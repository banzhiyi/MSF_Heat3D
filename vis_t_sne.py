import argparse
import inspect
import os
import random

import numpy as np
import torch

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，适合服务器运行
import matplotlib.pyplot as plt

from dataset import prepare_dataset
from vheat3d_model import MSF_Heat3D  # 按你项目实际模型文件名修改导入


# ============================================================
# 1. 参数配置
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Paper-style t-SNE visualization for HSI classification features.")

    # ---------------- Dataset ----------------
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--patches", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=64)

    # ---------------- Checkpoint ----------------
    p.add_argument("--ckpt", type=str, required=True)

    # ---------------- Feature collection ----------------
    p.add_argument("--samples_per_class", type=int, default=200)
    p.add_argument("--feat_stage", type=str, default="logits_pre",
                   choices=["initial", "logits_pre", "both"],
                   help="选择用于 t-SNE 的特征阶段：initial / logits_pre / both")
    p.add_argument("--label_base", type=str, default="zero",
                   choices=["zero", "one"],
                   help="标签编号方式。通常你的项目里应为 zero，即标签从 0 开始。")

    # ---------------- t-SNE parameters ----------------
    p.add_argument("--pca_dim", type=int, default=50,
                   help="t-SNE 前 PCA 预降维维度。设为 0 表示不使用 PCA。")
    p.add_argument("--perplexity", type=float, default=50,
                   help="t-SNE perplexity。若大于样本数限制，会自动调整。")
    p.add_argument("--early_exaggeration", type=float, default=16)
    p.add_argument("--tsne_iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)

    # ---------------- Plot style ----------------
    p.add_argument("--out", type=str, default="./results/vis_results/tsne_feat.png")
    p.add_argument("--fig_w", type=float, default=5.2)
    p.add_argument("--fig_h", type=float, default=4.0)
    p.add_argument("--point_size", type=float, default=16)
    p.add_argument("--alpha", type=float, default=0.90)
    p.add_argument("--dpi", type=int, default=600)

    p.add_argument("--show_axis", action="store_true",
                   help="是否显示坐标轴。默认不显示，更接近论文风格。")
    p.add_argument("--show_legend", action="store_true",
                   help="是否显示图例。默认不显示，更接近论文风格。")
    p.add_argument("--show_title", action="store_true",
                   help="是否显示标题。默认不显示。")
    p.add_argument("--title", type=str, default="",
                   help="自定义标题，例如 Method 或 Dataset name。")
    p.add_argument("--save_pdf", action="store_true",
                   help="是否同时保存 PDF 矢量图。")

    # ---------------- Model hyperparameters ----------------
    # 这些参数需要与你训练时保持一致。
    p.add_argument("--reduced_bands", type=int, default=24)
    p.add_argument("--heat_hidden_dim", type=int, default=48)
    p.add_argument("--head_channels", type=int, default=128)
    p.add_argument("--reducer_type", type=str, default="learnable")
    p.add_argument("--freq_pool", type=str, default="avgmax")
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--no_post_norm", action="store_true")
    p.add_argument("--no_multiscale", action="store_true")

    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


# ============================================================
# 2. 工具函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def split_out_path(out_path, suffix):
    stem, ext = os.path.splitext(out_path)
    ext = ext if ext else ".png"
    return f"{stem}_{suffix}{ext}"


def strip_module_prefix(state_dict):
    """
    兼容 DataParallel / DDP 保存的 checkpoint。
    """
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_paper_palette(num_classes):
    """
    高饱和、高区分度配色，适合 HSI 分类 t-SNE。
    如果类别数超过 20，则自动使用 tab20 扩展。
    """
    colors = [
        "#e41a1c",  # 1 red
        "#377eb8",  # 2 blue
        "#4daf4a",  # 3 green
        "#984ea3",  # 4 purple
        "#ff7f00",  # 5 orange
        "#ffff33",  # 6 yellow
        "#a65628",  # 7 brown
        "#f781bf",  # 8 pink
        "#999999",  # 9 gray
        "#1b9e77",  # 10 teal
        "#d95f02",  # 11 dark orange
        "#7570b3",  # 12 indigo
        "#66a61e",  # 13 olive green
        "#e7298a",  # 14 magenta
        "#a6761d",  # 15 mustard
        "#00bcd4",  # 16 cyan
        "#8dd3c7",  # 17 light cyan
        "#fb8072",  # 18 salmon
        "#80b1d3",  # 19 light blue
        "#b3de69",  # 20 light green
    ]

    if num_classes <= len(colors):
        return colors[:num_classes]

    cmap = plt.get_cmap("tab20", num_classes)
    return [cmap(i) for i in range(num_classes)]


def preprocess_for_tsne(X, seed=0, pca_dim=50):
    """
    推荐的 t-SNE 输入预处理流程：

    1. 将特征展平为二维矩阵；
    2. 处理 NaN / Inf；
    3. StandardScaler 标准化；
    4. PCA 预降维，提高 t-SNE 稳定性。
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X = StandardScaler().fit_transform(X)

    if pca_dim is not None and pca_dim > 0 and X.shape[1] > pca_dim:
        n_components = min(pca_dim, X.shape[0] - 1, X.shape[1])

        if n_components >= 2:
            X = PCA(n_components=n_components, random_state=seed).fit_transform(X)

    return X


def choose_perplexity(n_samples, user_perplexity):
    """
    perplexity 必须小于样本数。
    这里自动进行安全修正，避免 sklearn 报错。
    """
    if n_samples <= 5:
        raise ValueError(f"t-SNE 样本数过少：n_samples={n_samples}，请增大 samples_per_class。")

    max_valid = max(2, min(80, (n_samples - 1) / 3.0))

    if user_perplexity <= 0:
        return max_valid

    return min(float(user_perplexity), max_valid)


def run_tsne(X, seed, pca_dim, perplexity, early_exaggeration, tsne_iter):
    """
    执行 PCA + t-SNE。
    """
    X = preprocess_for_tsne(X, seed=seed, pca_dim=pca_dim)

    n_samples = X.shape[0]
    perplexity = choose_perplexity(n_samples, perplexity)

    tsne_kwargs = dict(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        metric="euclidean",
        method="barnes_hut",
        angle=0.5,
        verbose=1,
    )

    # 兼容不同 sklearn 版本
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = tsne_iter
    else:
        tsne_kwargs["n_iter"] = tsne_iter

    tsne = TSNE(**tsne_kwargs)
    Z = tsne.fit_transform(X)

    # 坐标居中并归一化，让图像更紧凑
    Z = Z - Z.mean(axis=0, keepdims=True)
    Z = Z / (np.max(np.abs(Z)) + 1e-8)

    return Z


def plot_tsne(
    X,
    y,
    num_classes,
    seed,
    out_path,
    title="",
    pca_dim=50,
    perplexity=50,
    early_exaggeration=16,
    tsne_iter=2000,
    fig_w=5.2,
    fig_h=4.0,
    point_size=16,
    alpha=0.90,
    dpi=600,
    show_axis=False,
    show_legend=False,
    show_title=False,
    save_pdf=False,
):
    """
    论文风格 t-SNE 绘图。
    """
    Z = run_tsne(
        X=X,
        seed=seed,
        pca_dim=pca_dim,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        tsne_iter=tsne_iter,
    )

    palette = get_paper_palette(num_classes)

    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for c in range(num_classes):
        idx = (y == c)

        if not np.any(idx):
            continue

        ax.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=point_size,
            c=palette[c],
            alpha=alpha,
            marker="o",
            edgecolors="none",
            linewidths=0,
            label=str(c + 1),
            rasterized=True,
        )

    if show_axis:
        ax.tick_params(axis="both", labelsize=9)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(False)

    if show_title:
        if title is None or title == "":
            title = "t-SNE"
        ax.set_title(title, fontsize=15, y=-0.14)

    if show_legend:
        ax.legend(
            loc="lower right",
            ncol=2,
            fontsize=7,
            frameon=False,
            markerscale=1.2,
            handletextpad=0.2,
            columnspacing=0.5,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.03)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    if save_pdf:
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        plt.savefig(
            pdf_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"saved: {pdf_path}")

    plt.close()

    print(f"saved: {out_path}")


# ============================================================
# 3. 特征收集
# ============================================================

@torch.no_grad()
def collect_features_per_class(
    model,
    loader,
    num_classes,
    samples_per_class,
    device,
    feat_stage="logits_pre",
    label_base="zero",
):
    """
    从测试集或验证集中按类别均衡采样特征。

    返回：
        X_by_stage: dict, 每个 stage 对应一个特征矩阵
        y: 标签向量，范围为 0 到 num_classes-1
    """
    model.eval()

    stages = [feat_stage] if feat_stage != "both" else ["initial", "logits_pre"]

    feats_by_stage = {
        stage: {c: [] for c in range(num_classes)}
        for stage in stages
    }

    done_classes = set()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        _, feat_out = model(
            xb,
            return_feat=True,
            feat_stage=feat_stage,
        )

        if feat_stage == "both":
            if not isinstance(feat_out, dict):
                raise RuntimeError(
                    "当 feat_stage='both' 时，模型应返回 dict 类型特征，例如 {'initial': ..., 'logits_pre': ...}。"
                )
            feat_map = {
                k: v.detach().cpu()
                for k, v in feat_out.items()
            }
        else:
            feat_map = {
                feat_stage: feat_out.detach().cpu()
            }

        yb_cpu = yb.detach().cpu().numpy()
        batch_size = next(iter(feat_map.values())).shape[0]

        for i in range(batch_size):
            c = int(yb_cpu[i])

            if label_base == "one":
                c = c - 1

            if c < 0 or c >= num_classes:
                continue

            if len(feats_by_stage[stages[0]][c]) >= samples_per_class:
                continue

            for stage in stages:
                feats_by_stage[stage][c].append(
                    feat_map[stage][i].numpy()
                )

            if len(feats_by_stage[stages[0]][c]) >= samples_per_class:
                done_classes.add(c)

        if len(done_classes) == num_classes:
            break

    X_by_stage = {}
    y_ref = None

    for stage in stages:
        X_list = []
        y_list = []

        for c in range(num_classes):
            arr = feats_by_stage[stage][c]

            if len(arr) == 0:
                print(f"[Warning] class {c + 1} has no collected samples.")
                continue

            X_list.append(np.stack(arr, axis=0))
            y_list.append(np.full((len(arr),), c, dtype=np.int64))

        if len(X_list) == 0:
            raise RuntimeError(f"未收集到任何 {stage} 特征，请检查数据集、标签和模型输出。")

        X_by_stage[stage] = np.concatenate(X_list, axis=0)
        y_cur = np.concatenate(y_list, axis=0)

        if y_ref is None:
            y_ref = y_cur

        print(f"\nCollected features for stage: {stage}")
        print(f"Feature shape: {X_by_stage[stage].shape}")

    print("\nCollected samples per class:")
    for c in range(num_classes):
        count = int(np.sum(y_ref == c))
        print(f"Class {c + 1}: {count}")

    return X_by_stage, y_ref


# ============================================================
# 4. 主函数
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # 4.1 准备数据集
    # --------------------------------------------------------
    class ArgsObj:
        pass

    data_args = ArgsObj()
    data_args.dataset = args.dataset
    data_args.data_root = args.data_root
    data_args.patches = args.patches
    data_args.batch_size = args.batch_size

    # 确保不会走 HybridSN 的 PCA 分支
    data_args.model_name = "Heat3D"

    _, label_test_loader, _, band, _, _, num_classes, _, _ = prepare_dataset(
        data_args,
        samples_type="ratio"
    )

    print(f"Dataset: {args.dataset}")
    print(f"Band number: {band}")
    print(f"Class number: {num_classes}")

    # --------------------------------------------------------
    # 4.2 构建模型
    # --------------------------------------------------------
    model = MSF_Heat3D(
        band=band,
        num_classes=num_classes,
        patches=args.patches,
        reduced_bands=args.reduced_bands,
        heat_hidden_dim=args.heat_hidden_dim,
        head_channels=args.head_channels,
        reducer_type=args.reducer_type,
        use_checkpoint=args.use_checkpoint,
        freq_pool=args.freq_pool,
        use_post_norm=not args.no_post_norm,
        use_multiscale=not args.no_multiscale,
        dataset_name=args.dataset.lower(),
    ).to(device)

    model.eval()

    # --------------------------------------------------------
    # 4.3 加载 checkpoint
    # --------------------------------------------------------
    print(f"Loading checkpoint from: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "net" in ckpt:
            state = ckpt["net"]
        else:
            state = ckpt
    else:
        state = ckpt

    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if len(missing) > 0:
        print(f"[Warning] missing keys: {len(missing)}")
        for k in missing[:20]:
            print(f"  missing: {k}")
        if len(missing) > 20:
            print("  ...")

    if len(unexpected) > 0:
        print(f"[Warning] unexpected keys: {len(unexpected)}")
        for k in unexpected[:20]:
            print(f"  unexpected: {k}")
        if len(unexpected) > 20:
            print("  ...")

    # --------------------------------------------------------
    # 4.4 收集特征
    # --------------------------------------------------------
    X_by_stage, y = collect_features_per_class(
        model=model,
        loader=label_test_loader,
        num_classes=num_classes,
        samples_per_class=args.samples_per_class,
        device=device,
        feat_stage=args.feat_stage,
        label_base=args.label_base,
    )

    # --------------------------------------------------------
    # 4.5 绘制 t-SNE
    # --------------------------------------------------------
    stage_titles = {
        "initial": "Initial Features",
        "logits_pre": "Pre-logits Features",
    }

    if args.feat_stage == "both":
        for stage, X in X_by_stage.items():
            if args.title != "":
                cur_title = f"{args.title}-{stage}"
            else:
                cur_title = stage_titles.get(stage, stage)

            out_path = split_out_path(args.out, stage)

            plot_tsne(
                X=X,
                y=y,
                num_classes=num_classes,
                seed=args.seed,
                out_path=out_path,
                title=cur_title,
                pca_dim=args.pca_dim,
                perplexity=args.perplexity,
                early_exaggeration=args.early_exaggeration,
                tsne_iter=args.tsne_iter,
                fig_w=args.fig_w,
                fig_h=args.fig_h,
                point_size=args.point_size,
                alpha=args.alpha,
                dpi=args.dpi,
                show_axis=args.show_axis,
                show_legend=args.show_legend,
                show_title=args.show_title,
                save_pdf=args.save_pdf,
            )
    else:
        stage = args.feat_stage
        X = X_by_stage[stage]

        if args.title != "":
            cur_title = args.title
        else:
            cur_title = stage_titles.get(stage, stage)

        plot_tsne(
            X=X,
            y=y,
            num_classes=num_classes,
            seed=args.seed,
            out_path=args.out,
            title=cur_title,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            early_exaggeration=args.early_exaggeration,
            tsne_iter=args.tsne_iter,
            fig_w=args.fig_w,
            fig_h=args.fig_h,
            point_size=args.point_size,
            alpha=args.alpha,
            dpi=args.dpi,
            show_axis=args.show_axis,
            show_legend=args.show_legend,
            show_title=args.show_title,
            save_pdf=args.save_pdf,
        )


if __name__ == "__main__":
    main()