import os

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端：只保存图片

import matplotlib.pyplot as plt


# 1) 把不同数据集的数据集中到这里:
#    结构: dataset_name -> method_name -> list[OA at 20,40,60,80,100]
ALL_DATA = {
    "Indian": {
        "2D-CNN": [42.09, 50.33, 67.31, 78.54, 83.69],
        "3D-CNN": [53.13, 68.04, 78.46, 83.22, 86.55],
        "HybridSN": [51.97, 64.57, 79.38, 80.04, 89.46],
        "ViT": [48.11, 64.24, 75.87, 81.22, 86.48],
        "MorphFormer": [57.14, 82.91, 90.09, 90.84, 97.40],
        "SSFTT": [68.23, 80.67, 86.27, 89.27, 93.71],
        "VMamba": [55.77, 71.61, 80.05, 82.09, 86.32],
        "GraphMamba": [61.36, 78.07, 87.21, 91.30, 96.58],
        "S2Mamba": [52.73, 75.50, 86.62, 89.28, 95.04],
        "vHeat": [62.87, 80.13, 87.35, 88.35, 95.55],
        "MSFHeat3D": [65.94, 83.17, 90.45, 92.12, 97.56],
    },
    "Augsburg": {
        "2D-CNN": [73.11, 77.64, 79.81, 83.12, 84.44],
        "3D-CNN": [74.65, 82.14, 85.29, 85.63, 87.28],
        "HybridSN": [64.52, 73.40, 77.31, 85.21, 86.22],
        "ViT": [77.53, 78.71, 78.64, 82.65, 86.12],
        "MorphFormer": [83.29, 86.01, 87.18, 89.71, 90.24],
        "SSFTT": [84.38, 87.66, 89.21, 90.72, 91.53],
        "VMamba": [83.50, 84.43, 84.80, 87.58, 87.78],
        "GraphMamba": [84.49, 86.22, 88.63, 90.83, 91.08],
        "S2Mamba": [83.32, 87.47, 88.11, 90.79, 91.23],
        "vHeat": [84.14, 87.24, 87.82, 88.70, 90.38],
        "MSFHeat3D": [84.53, 87.79, 89.33, 90.88, 91.77],
    },
    "Houston": {
        "2D-CNN": [68.18, 80.41, 84.60, 87.58, 88.78],
        "3D-CNN": [68.54, 75.84, 81.48, 86.42, 88.24],
        "HybridSN": [71.52, 77.68, 82.10, 85.99, 88.14],
        "ViT": [67.79, 79.89, 83.38, 87.15, 88.80],
        "MorphFormer": [74.72, 81.94, 88.81, 90.86, 93.29],
        "SSFTT": [62.37, 79.41, 84.59, 88.18, 90.39],
        "VMamba": [72.61, 81.82, 83.76, 86.15, 88.93],
        "GraphMamba": [58.69, 82.80, 84.67, 90.36, 90.64],
        "S2Mamba": [75.11, 82.97, 88.44, 91.43, 93.87],
        "vHeat": [70.78, 82.74, 88.82, 90.93, 92.70],
        "MSFHeat3D": [76.15, 84.80, 88.97, 91.69, 95.30],
    },
}

# 2) 保证 11 种方法顺序固定（也保证 marker 对应固定）
METHODS = [
    "2D-CNN",
    "3D-CNN",
    "HybridSN",
    "ViT",
    "MorphFormer",
    "SSFTT",
    "VMamba",
    "GraphMamba",
    "S2Mamba",
    "vHeat",
    "MSFHeat3D",
]

# 所有方法用实线，用不同 marker 区分
MARKERS = ["o", "s", "D", "^", "v", ">", "<", "p", "h", "X", "*"]
# 不同数据集的 y 轴范围配置
Y_AXIS_CFG = {
    "Indian": (40, 100),
    "Augsburg": (60, 95),
    "Houston": (55, 100),
}


def plot_dataset(dataset_name: str, data: dict, out_dir: str) -> str:
    x = [20, 40, 60, 80, 100]

    plt.figure(figsize=(11, 6.5), dpi=150)

    for method, mk in zip(METHODS, MARKERS):
        if method not in data:
            raise KeyError(f"{dataset_name} 缺少方法数据: {method}")

        y = data[method]
        if len(y) != len(x):
            raise ValueError(f"{dataset_name}/{method} 的数据长度应为 {len(x)}，但得到 {len(y)}")

        kwargs = dict(
            linestyle="-",  # 统一实线
            marker=mk,
            linewidth=2.0,
            markersize=6,
            label=method,
        )
        if method == "MSFHeat3D":
            kwargs["color"] = "darkred"  # 深红色

        plt.plot(x, y, **kwargs)

    plt.xlabel("Training ratio (%)", fontsize=16)
    plt.ylabel("Overall Accuracy (%)", fontsize=16)
    plt.xticks(x, x)
    # y 轴：按数据集设置范围（并生成对应刻度）
    y_min, y_max = Y_AXIS_CFG.get(dataset_name, (30, 100))
    plt.ylim(y_min, y_max)

    # 生成 10 为步长的刻度；若上限不是 10 的倍数（如 95）也会包含进去
    ticks = list(range(int((y_min + 9) // 10 * 10), int(y_max // 10 * 10) + 1, 10))
    if y_min % 10 == 0:
        ticks = [y_min] + ticks
    if ticks and ticks[-1] != y_max:
        ticks.append(y_max)
    elif not ticks:
        ticks = [y_min, y_max]
    plt.yticks(ticks)

    plt.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.legend(
        loc="lower right",
        frameon=True,
        fontsize=14,
        ncol=1,
        borderpad=0.6,
        handlelength=2.6,
    )

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_OA_training_ratio.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return out_path


def main() -> None:
    out_dir = os.path.join(".", "results", "vis_results")

    for dataset_name, data in ALL_DATA.items():
        if not data:
            continue

        out_path = plot_dataset(dataset_name, data, out_dir)
        print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()


