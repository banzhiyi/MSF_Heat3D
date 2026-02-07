import argparse
import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")  # 强制使用无GUI后端，避免Tk工具栏/图标相关错误
import matplotlib.pyplot as plt

from dataset import prepare_dataset
from vheat3d_model import MSF_Heat3D  # 按你项目实际模型文件名修改导入

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--patches", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--samples_per_class", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="./results/vis_results/tsne_feat.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

@torch.no_grad()
def collect_features_per_class(model, loader, num_classes, samples_per_class, device):
    feats_by_class = {c: [] for c in range(num_classes)}
    done = set()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # dataset 输出是 (B, C, H, W)，模型 _to_channels_last_hw_s 会自动识别 band 维
        logits, feat_vec = model(xb, return_feat=True)  # feat_vec: (B, D)

        feat_vec = feat_vec.detach().cpu()
        yb_cpu = yb.detach().cpu().numpy()

        for i in range(feat_vec.shape[0]):
            c = int(yb_cpu[i])
            if c < 0 or c >= num_classes:
                continue
            if len(feats_by_class[c]) >= samples_per_class:
                continue
            feats_by_class[c].append(feat_vec[i].numpy())
            if len(feats_by_class[c]) >= samples_per_class:
                done.add(c)

        if len(done) == num_classes:
            break

    X = []
    y = []
    for c in range(num_classes):
        arr = feats_by_class[c]
        if len(arr) == 0:
            continue
        X.append(np.stack(arr, axis=0))
        y.append(np.full((len(arr),), c, dtype=np.int64))

    if len(X) == 0:
        raise RuntimeError("未收集到任何特征，请检查数据集与标签。")

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 复用你现有数据准备
    class ArgsObj:
        pass
    a = ArgsObj()
    a.dataset = args.dataset
    a.patches = args.patches
    a.batch_size = args.batch_size
    a.model_name = "Heat3D"  # 确保不会走 HybridSN 的 PCA 分支

    _, label_test_loader, _, band, _, _, num_classes, _, _ = prepare_dataset(a, samples_type="ratio")

    # 构建模型（按你训练时的超参保持一致）
    model = MSF_Heat3D(
        band=band,
        num_classes=num_classes,
        patches=args.patches,
        reduced_bands=24,
        heat_hidden_dim=48,
        head_channels=128,
        reducer_type="learnable",
        use_checkpoint=False,
        freq_pool="avgmax",
        use_post_norm=True,
        use_multiscale=True,
        dataset_name=args.dataset.lower(),
    ).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"load\_state\_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    X, y = collect_features_per_class(
        model=model,
        loader=label_test_loader,
        num_classes=num_classes,
        samples_per_class=args.samples_per_class,
        device=device,
    )

    tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto", perplexity=30)
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for c in range(num_classes):
        idx = (y == c)
        if not np.any(idx):
            continue
        plt.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=0.8, label=str(c + 1))
    plt.legend(markerscale=2, loc="upper right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()