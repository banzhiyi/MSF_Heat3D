import os
import argparse
from typing import Optional

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


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

#indian数据集使用这个build_fasle_color函数效果更好
def build_false_color(hsi: np.ndarray, bands=(30, 20, 10)):
    b0, b1, b2 = [min(idx, hsi.shape[2] - 1) for idx in bands]
    rgb = np.stack([hsi[:, :, b0], hsi[:, :, b1], hsi[:, :, b2]], axis=-1)
    rgb_min = rgb.min(axis=(0, 1), keepdims=True)
    rgb_max = rgb.max(axis=(0, 1), keepdims=True)
    denom = np.clip(rgb_max - rgb_min, a_min=1e-6, a_max=None)
    return np.clip((rgb - rgb_min) / denom, 0.0, 1.0)

#其余数据集使用这个build_false_color函数效果更好
"""
def build_false_color(hsi: np.ndarray, bands=(30, 20, 10), p_low: float = 2.0, p_high: float = 98.0):
    b0, b1, b2 = [min(idx, hsi.shape[2] - 1) for idx in bands]
    rgb = np.stack([hsi[:, :, b0], hsi[:, :, b1], hsi[:, :, b2]], axis=-1).astype(np.float32)

    lo = np.percentile(rgb, p_low, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, p_high, axis=(0, 1), keepdims=True)
    denom = np.clip(hi - lo, a_min=1e-6, a_max=None)

    rgb = (rgb - lo) / denom
    return np.clip(rgb, 0.0, 1.0)
"""


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
    tr = sio.loadmat(parts["train"])["TrainImage"].astype(np.int32)
    te = sio.loadmat(parts["test"])["TestImage"].astype(np.int32)
    gt = np.where(tr != 0, tr, te)
    num_classes = int(gt.max())
    if num_classes <= 0:
        raise ValueError(f"{root_dir} 未检测到有效类别。")
    return hsi, tr, te, gt, num_classes


def load_single_mat_dataset(mat_path: str):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"{mat_path} 不是有效的 .mat 文件")
    data = sio.loadmat(mat_path)
    hsi = normalize_hsi(data["input"].astype(np.float32))
    tr = data["TR"].astype(np.int32)
    te = data["TE"].astype(np.int32)
    gt = np.where(tr != 0, tr, te)
    num_classes = int(gt.max())
    if num_classes <= 0:
        raise ValueError(f"{mat_path} 未检测到有效类别。")
    return hsi, tr, te, gt, num_classes


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



def build_class_colormap(num_classes: int, background_color=(0, 0, 0), class_colors=None):
    colors = [background_color]
    if class_colors is not None and len(class_colors) >= num_classes:
        colors.extend(class_colors[:num_classes].tolist())
    else:
        colors.extend(plt.cm.get_cmap("tab20", num_classes).colors[:num_classes])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, num_classes + 1.5, 1.0), cmap.N)
    return cmap, norm


def save_image(fig_data: np.ndarray, title: str, out_path: str, cmap=None, norm=None,show_title: bool = False):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(fig_data, cmap=cmap, norm=norm)
    plt.axis("off")
    if show_title and title:
        plt.title(title)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize HSI: false color and GT only.")
    parser.add_argument("--data_path", default="./data/IndianPine.mat")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--output_dir", default="./results/vis_results")
    parser.add_argument("--false_color_bands", nargs=3, type=int, default=(30, 20, 10))
    args = parser.parse_args()

    dataset_key = infer_dataset_key(args.data_path, args.dataset_name)
    dataset_name = (args.dataset_name or dataset_key.capitalize()).replace(".mat", "")
    safe_name = dataset_name.lower().replace(" ", "_")
    title_prefix = dataset_name.replace("_", " ")

    hsi, _tr, _te, gt, num_classes = load_hsi_dataset(args.data_path, dataset_key)
    false_color = build_false_color(hsi, bands=tuple(args.false_color_bands))

    class_colors = select_class_colors(dataset_key)
    cmap, norm = build_class_colormap(num_classes, class_colors=class_colors)

    save_image(
        false_color,
        f"{title_prefix} False Color",
        os.path.join(args.output_dir, f"{safe_name}_false_color.png"),
        cmap=None,
        norm=None,
    )
    save_image(
        gt,
        f"{title_prefix} Ground Truth",
        os.path.join(args.output_dir, f"{safe_name}_gt.png"),
        cmap=cmap,
        norm=norm,
    )
    print(f"saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
