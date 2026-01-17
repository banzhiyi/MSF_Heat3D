import os
import argparse
from typing import Optional
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn.functional as F


from vHeat import S2VHeat
from vheat3d_model import MSF_Heat3D
from DSNet import DSNet
from MASSFormer import MASSFormer
from SiT import SiT
from HSI2DCNN import HSI2DCNN
from vHeat import S2VHeat
from VisionMamba.VMamba import VisionMambaClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INDIAN_PINES_COLORS = np.array(
    [[c / 256.0 for c in color] for color in [
        [83, 171, 72],#1 背景颜色为黑色，会被build_class_colormap()自动添加
        [137, 186, 67],#2
        [66, 132, 91],#3
        [60, 131, 69],#4
        [144, 82, 54],#5
        [105, 188, 200],#6
        [255, 255, 255],#7
        [199, 176, 201],#8
        [218, 51, 44],#9
        [119, 35, 36],#10
        [55, 101, 166],#11
        [224, 219, 84],#12
        [217, 142, 52],#13
        [84, 48, 126],#14
        [227, 119, 91],#15
        [157, 87, 150],#16
    ]],
    dtype=np.float32,
)

AUGSBURG_COLORS = np.array(
    [[c / 256.0 for c in color] for color in [
        [60, 131, 69],#1
        [218, 51, 44],#2
        [224, 219, 84],#3
        [137, 186, 67],#4
        [83, 171, 72],#5
        [105, 188, 200],#6
        [55, 101, 166],#7
    ]],
    dtype=np.float32,
)

PAVIA_COLORS = np.array(
    [[c / 256.0 for c in color] for color in [
        [83, 171, 72],#1
        [66, 132, 91],#2
        [144, 82, 54],#3
        [255, 255, 255],#4
        [218, 51, 44],#5
        [55, 101, 166],#6
        [217, 142, 52],#7
        [227, 119, 91],#8
        [157, 87, 150],#9
    ]],
    dtype=np.float32,
)

HOUSTON_COLORS = np.array(
    [[c / 256.0 for c in color] for color in [
        [83, 171, 72],#1
        [137, 186, 67],#2
        [66, 132, 91],#3
        [60, 131, 69],#4
        [144, 82, 54],#5
        [105, 188, 200],#6
        [255, 255, 255],#7
        [218, 51, 44],#8
        [119, 35, 36],#9
        [55, 101, 166],#10
        [224, 219, 84],#11
        [217, 142, 52],#12
        [84, 48, 126],#13
        [227, 119, 91],#14
        [157, 87, 150],#15
    ]],
    dtype=np.float32,
)

DATASET_ALIASES = {
    "indian": ["indian", "indianpines", "indianpine"],
    "pavia": ["pavia"],
    "houston": ["houston"],
    "augsburg": ["augsburg"],
}

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
    if dataset_key in {"augsburg"}:
        root_dir = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)
        root_dir = root_dir or "."
        return load_split_mat_dataset(root_dir)
    return load_single_mat_dataset(data_path)

def select_class_colors(dataset_key: str):
    dataset_key = (dataset_key or "").lower()
    if dataset_key == "augsburg":
        return AUGSBURG_COLORS
    if dataset_key == "pavia":
        return PAVIA_COLORS
    if dataset_key == "houston":
        return HOUSTON_COLORS
    # 默认 indian
    return INDIAN_PINES_COLORS

def build_model(
    band: int,
    num_classes: int,
    patch_size: int,
    ckpt_path: str,
    *,
    model_name: str = "MSF_Heat3D",
    reduced_bands=24,
    heat_hidden_dim=64,
    head_channels=128,
    reducer_type="learnable",
    pca_path: Optional[str] = None,
    freq_pool="avgmax",
    use_post_norm=True,
    dataset_name: str,
):
    """
    返回一个可直接前向输出 logits 的模型（B, num_classes）。
    """
    model_name = (model_name or "").lower()

    if model_name in {"dsnet"}:
        # DSNet: 你的 DSNet.py 已经兼容 (band, num_classes, patch_size)
        model = DSNet(band, num_classes, patch_size)
    elif model_name in {"massformer"}:
        # 接口与 demo.py 对齐：MASSFormer(band, num\_classes, patch\_size)
        model = MASSFormer(band, num_classes, patch_size)
    elif model_name == "sit":
        model = SiT(band, num_classes, patch_size)
    elif model_name == "hsi2dcnn":
        model = HSI2DCNN(band, num_classes, patch_size)
    elif model_name == "vheat":
        model = S2VHeat(band, num_classes, patch_size)
    elif model_name == "vmamba":
        model = VisionMambaClassifier(
            band=band,
            num_classes=num_classes,
            img_size=patch_size,  # 与当前 patch 裁剪尺寸一致
            embed_dim=128,
            depth=8,
            d_state=8,
            drop_path_rate=0.6,
            if_abs_pos_embed=True,
            if_rope=False,
            if_cls_token=True,
            use_middle_cls_token=True,
        )
    elif model_name in {"msf_heat3d", "msf-heat3d", "heat3d", "vheat3d"}:
        pca_tensor = None
        if reducer_type == "pca":
            # 若你原本这里有 PCA 加载逻辑，保持不变
            pass

        model = MSF_Heat3D(
            band,
            num_classes,
            patch_size,
            dataset_name=dataset_name,
        )
    else:
        raise KeyError(f"unknown model_name: {model_name}")

    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state, strict=True)
    else:
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

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
    """
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
    """

    def flush_batch():
        if not patches:
            return

        # patches: List[np.ndarray]，每个元素形状应为 (patch_size, patch_size, band)
        batch_np = np.stack(patches, axis=0).astype(np.float32)  # (B, H, W, C)
        batch_np = batch_np.transpose(0, 3, 1, 2).copy()  # (B, C, H, W)

        batch = torch.from_numpy(batch_np).to(device)

        with torch.no_grad():
            logits = model(batch)  # (B, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()

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

def save_pred_only(pred: np.ndarray, dataset_name: str, save_dir: str, num_classes: int, class_colors):
    os.makedirs(save_dir, exist_ok=True)
    safe_name = dataset_name.lower().replace(" ", "_")
    # title_prefix = dataset_name.replace("_", " ")  # \u5982\u679c\u4e0d\u7528\u6807\u9898\uff0c\u53ef\u4ee5\u4e0d\u9700\u8981

    cmap, norm = build_class_colormap(num_classes, class_colors=class_colors)

    plt.figure(figsize=(5, 5))
    plt.imshow(pred, cmap=cmap, norm=norm)
    plt.axis("off")
    # plt.title(f"{title_prefix} Heat3D_Pipeline Prediction")  # \u5220\u9664\u6216\u6ce8\u91ca\uff0c\u53bb\u6389\u6807\u9898
    plt.savefig(os.path.join(save_dir, f"{safe_name}_pred.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved pred figure to {save_dir}")




def main():
    parser = argparse.ArgumentParser(description="Visualize HSI predictions.")
    parser.add_argument("--data_path", default="./data/IndianPine.mat")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--patch_size", type=int, default=9)
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
    parser.add_argument("--model_name", choices=["MSF_Heat3D", "DSNet", "MASSFormer", "SiT", "HSI2DCNN", "vHeat", "VMamba"], default="MSF_Heat3D")
    args = parser.parse_args()

    dataset_key = infer_dataset_key(args.data_path, args.dataset_name)
    dataset_name = (args.dataset_name or dataset_key.capitalize()).replace(".mat", "")
    hsi, TR, TE, gt, num_classes = load_hsi_dataset(args.data_path, dataset_key)

    class_colors = select_class_colors(dataset_key)
    model = build_model(
        band=hsi.shape[2],
        num_classes=num_classes,
        patch_size=args.patch_size,
        ckpt_path=args.ckpt_path,
        reduced_bands=args.reduced_bands,
        model_name=args.model_name,
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
    pred_vis =  pred_map + 1
    save_pred_only(pred_vis, dataset_name,
                 args.output_dir, num_classes, class_colors, )

if __name__ == "__main__":
    main()
