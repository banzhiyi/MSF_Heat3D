import os
import argparse
from typing import Optional
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn.functional as F
from vheat3d_model import MSF_Heat3D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INDIAN_PINES_COLORS = np.array([
    [0.24, 0.52, 0.20], [0.51, 0.73, 0.26], [0.20, 0.40, 0.14], [0.42, 0.61, 0.28],
    [0.41, 0.35, 0.21], [0.40, 0.77, 0.86], [0.50, 0.39, 0.60], [0.73, 0.65, 0.79],
    [0.78, 0.23, 0.19], [0.75, 0.64, 0.14], [0.22, 0.35, 0.63], [0.63, 0.68, 0.28],
    [0.30, 0.20, 0.13], [0.86, 0.36, 0.17], [0.79, 0.46, 0.35], [0.55, 0.28, 0.55],
], dtype=np.float32)

BERLIN_AUGSBURG_COLORS = np.array([
    [0.88, 0.17, 0.17], [0.10, 0.55, 0.85], [0.96, 0.78, 0.20], [0.31, 0.74, 0.46],
    [0.56, 0.34, 0.74], [0.92, 0.53, 0.20], [0.17, 0.62, 0.60], [0.60, 0.60, 0.60],
], dtype=np.float32)

DATASET_ALIASES = {
    "indian": ["indian", "indianpines", "indianpine"],
    "pavia": ["pavia"],
    "houston": ["houston"],
    "berlin": ["berlin"],
    "augsburg": ["augsburg"],
}

ZERO_BASE_PRED_DATASETS = {"berlin"}

def normalize_hsi(hsi: np.ndarray):
    band_min = hsi.min(axis=(0, 1), keepdims=True)
    band_max = hsi.max(axis=(0, 1), keepdims=True)
    denom = np.clip(band_max - band_min, a_min=1e-6, a_max=None)
    return np.clip((hsi - band_min) / denom, 0.0, 1.0)

def build_false_color(hsi: np.ndarray, bands=(30, 20, 10)):
    b0, b1, b2 = [min(idx, hsi.shape[2] - 1) for idx in bands]
    rgb = np.stack([hsi[:, :, b0], hsi[:, :, b1], hsi[:, :, b2]], axis=-1)
    rgb_min = rgb.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb.max(axis=(0, 1), keepdims=True)
    denom = np.clip(rgb_max - rgb_min, a_min=1e-6, a_max=None)
    return np.clip((rgb - rgb_min) / denom, 0.0, 1.0)

def infer_dataset_key(data_path: str, dataset_hint: Optional[str] = None) -> str:
    candidates = ([dataset_hint] if dataset_hint else []) + os.path.normpath(data_path).split(os.sep)
    candidates.append(os.path.splitext(os.path.basename(data_path))[0])
    for cand in candidates:
        if not cand:
            continue
        name = cand.lower()
        for key, aliases in DATASET_ALIASES.items():
            if any(alias in name for alias in aliases):
                return key
    return "indian"

def load_split_mat_dataset(root_dir: str):
    parts = {
        "cube": os.path.join(root_dir, "data_HS_LR.mat"),
        "train": os.path.join(root_dir, "TrainImage.mat"),
        "test": os.path.join(root_dir, "TestImage.mat"),
    }
    for label, path in parts.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} 文件缺失: {path}")
    hsi = normalize_hsi(sio.loadmat(parts["cube"])["data_HS_LR"].astype(np.float32))
    TR = sio.loadmat(parts["train"])["TrainImage"].astype(np.int32)
    TE = sio.loadmat(parts["test"])["TestImage"].astype(np.int32)
    gt = np.where(TR != 0, TR, TE)
    num_classes = int(gt.max())
    if num_classes <= 0:
        raise ValueError(f"{root_dir} 未检测到有效类别。")
    return hsi, TR, TE, gt, num_classes

def load_single_mat_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"{mat_path} 不是有效的 .mat 文件")
    data = sio.loadmat(mat_path)
    hsi = normalize_hsi(data["input"].astype(np.float32))
    TR = data["TR"].astype(np.int32)
    TE = data["TE"].astype(np.int32)
    gt = np.where(TR != 0, TR, TE)
    num_classes = int(gt.max())
    if num_classes <= 0:
        raise ValueError(f"{mat_path} 未检测到有效类别。")
    return hsi, TR, TE, gt, num_classes

def load_hsi_dataset(data_path: str, dataset_key: Optional[str] = None):
    dataset_key = (dataset_key or infer_dataset_key(data_path)).lower()
    if dataset_key in {"berlin", "augsburg"}:
        root_dir = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)
        root_dir = root_dir or "."
        return load_split_mat_dataset(root_dir)
    return load_single_mat_dataset(data_path)

def select_class_colors(dataset_key: str):
    if dataset_key in {"berlin", "augsburg"}:
        return BERLIN_AUGSBURG_COLORS
    return INDIAN_PINES_COLORS

def build_model(band: int, num_classes: int, patch_size: int, ckpt_path: str, *,
                reduced_bands=24, heat_hidden_dim=64,
                head_channels=128, reducer_type="learnable", pca_path: Optional[str] = None,
                freq_pool="avgmax", use_post_norm=True,dataset_name: str):
    pca_tensor = None
    if reducer_type == "pca":
        if not pca_path:
            raise ValueError("PCA 模式需要提供 --pca_path。")
        pca_tensor = torch.from_numpy(np.load(pca_path)).float()
    model = MSF_Heat3D(
        band=band,
        num_classes=num_classes,
        patches=patch_size,
        reduced_bands=24,
        heat_hidden_dim=48,
        head_channels=128,
        reducer_type="learnable",
        pca_P=pca_tensor,
        use_checkpoint=False,
        freq_pool=freq_pool,
        use_post_norm=use_post_norm,
        use_multiscale=True,  # \* 根据你当前 Heat3D_Pipeline 默认使用多尺度
        dataset_name=dataset_name,
    )
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"loaded checkpoint from {ckpt_path}")
    else:
        print(f"warning: checkpoint {ckpt_path} not found, use randomly initialized weights")
    model.eval()
    model.to(DEVICE)
    return model

def sliding_window_predict(hsi: np.ndarray, model: torch.nn.Module, patch_size: int,
                           num_classes: int, band: int, batch_size: int = 512):
    H, W, S = hsi.shape
    assert S == band, f"band mismatch: HSI has {S}, model expects {band}"
    pad = patch_size // 2
    hsi_pad = np.pad(hsi, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    pred_score_sum = np.zeros((H, W, num_classes), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)
    device = next(model.parameters()).device
    patches, coords = [], []

    def flush_batch():
        if not patches:
            return
        batch = torch.from_numpy(np.stack(patches).astype(np.float32)).to(device)
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1).cpu().numpy()
        for (h_idx, w_idx), prob in zip(coords, probs):
            pred_score_sum[h_idx, w_idx] += prob
            pred_count[h_idx, w_idx] += 1.0
        patches.clear()
        coords.clear()

    for i in range(H):
        for j in range(W):
            i_pad, j_pad = i + pad, j + pad
            patch = hsi_pad[i_pad - pad:i_pad + pad + 1, j_pad - pad:j_pad + pad + 1, :]
            if patch.shape[:2] != (patch_size, patch_size):
                continue
            patches.append(patch)
            coords.append((i, j))
            if len(patches) == batch_size:
                flush_batch()
    flush_batch()
    avg_scores = pred_score_sum / np.clip(pred_count[:, :, None], a_min=1.0, a_max=None)
    return np.argmax(avg_scores, axis=-1).astype(np.int32)

def build_class_colormap(num_classes: int, background_color=(0, 0, 0), class_colors=None):
    colors = [background_color]
    if class_colors is not None and len(class_colors) >= num_classes:
        colors.extend(class_colors[:num_classes].tolist())
    else:
        colors.extend(plt.cm.get_cmap("tab20", num_classes).colors[:num_classes])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_classes + 1.5, 1.0), cmap.N)
    return cmap, norm

def plot_results(false_color: np.ndarray, gt: np.ndarray, pred: np.ndarray, dataset_name: str,
                 save_dir: str, num_classes: int, class_colors, ):
    os.makedirs(save_dir, exist_ok=True)
    safe_name = dataset_name.lower().replace(" ", "_")
    title_prefix = dataset_name.replace("_", " ")
    cmap, norm = build_class_colormap(num_classes, class_colors=class_colors)
    ticks = np.arange(0, num_classes + 1)
    tick_labels = ["BG"] + [str(i) for i in range(1, num_classes + 1)]

    def _save(fig_data, title, filename, use_colorbar=False):
        plt.figure(figsize=(5, 5))
        im = plt.imshow(
            fig_data if fig_data.ndim == 3 else fig_data,
            cmap=None if fig_data.ndim == 3 else cmap,
            norm=None if fig_data.ndim == 3 else norm,
        )
        if use_colorbar:
            # 宽 >= 高: 认为是横着的长条，把颜色条放在下方
            is_wide = fig_data.shape[1] >= fig_data.shape[0]
            if is_wide:
                # 颜色条横向，放在下方
                cbar = plt.colorbar(
                    im,
                    orientation="horizontal",
                    fraction=0.046,
                    pad=0.12,
                    ticks=ticks,
                )
                cbar.ax.set_xticklabels(tick_labels)
            else:
                # 颜色条纵向，放在右侧
                cbar = plt.colorbar(
                    im,
                    orientation="vertical",
                    fraction=0.046,
                    pad=0.04,
                    ticks=ticks,
                )
                cbar.ax.set_yticklabels(tick_labels)
        plt.axis("off")
        plt.title(title)
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

    _save(false_color, f"{title_prefix} False Color", f"{safe_name}_false_color.png")
    _save(gt, f"{title_prefix} Ground Truth", f"{safe_name}_gt.png", use_colorbar=True)
    _save(pred, f"{title_prefix} Heat3D_Pipeline Prediction", f"{safe_name}_pred.png", use_colorbar=True)
    print(f"saved figures to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize HSI predictions.")
    parser.add_argument("--data_path", default="./data/IndianPine.mat")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--output_dir", default="./results/vis_results")
    parser.add_argument("--false_color_bands", nargs=3, type=int, default=(30, 20, 10))
    parser.add_argument("--reduced_bands", type=int, default=24)
    parser.add_argument("--heat_hidden_dim", type=int, default=64)
    parser.add_argument("--head_channels", type=int, default=128)
    parser.add_argument("--reducer_type", choices=["learnable", "pca"], default="learnable")
    parser.add_argument("--pca_path", default=None)
    parser.add_argument("--freq_pool", choices=["avg", "max", "avgmax"], default="avgmax")
    parser.add_argument("--disable_post_norm", action="store_true")
    args = parser.parse_args()

    dataset_key = infer_dataset_key(args.data_path, args.dataset_name)
    dataset_name = (args.dataset_name or dataset_key.capitalize()).replace(".mat", "")
    hsi, TR, TE, gt, num_classes = load_hsi_dataset(args.data_path, dataset_key)
    false_color = build_false_color(hsi, bands=tuple(args.false_color_bands))
    class_colors = select_class_colors(dataset_key)
    zero_based_pred = dataset_key in ZERO_BASE_PRED_DATASETS
    model = build_model(
        band=hsi.shape[2],
        num_classes=num_classes,
        patch_size=args.patch_size,
        ckpt_path=args.ckpt_path,
        reduced_bands=args.reduced_bands,
        heat_hidden_dim=args.heat_hidden_dim,
        head_channels=args.head_channels,
        reducer_type=args.reducer_type,
        pca_path=args.pca_path,
        freq_pool=args.freq_pool,
        use_post_norm=not args.disable_post_norm,
        dataset_name=dataset_name,
    )
    pred_map = sliding_window_predict(
        hsi=hsi,
        model=model,
        patch_size=args.patch_size,
        num_classes=num_classes,
        band=hsi.shape[2],
        batch_size=args.batch_size,
    )
    pred_vis = pred_map if zero_based_pred else pred_map + 1
    plot_results(false_color, gt, pred_vis,  dataset_name,
                 args.output_dir, num_classes, class_colors, )

if __name__ == "__main__":
    main()
