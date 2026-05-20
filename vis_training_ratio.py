import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 1. Data
# ============================================================

ALL_DATA = {
    "Indian": {
        "2D-CNN": [42.09, 50.33, 67.31, 78.54, 83.69],
        "3D-CNN": [53.13, 68.04, 78.46, 83.22, 86.55],
        "HybridSN": [51.97, 64.57, 79.38, 80.04, 89.46],
        "ViT": [48.11, 64.24, 75.87, 81.22, 86.48],
        "MorphFormer": [57.14, 82.91, 90.09, 90.84, 97.40],
        "SSFTT": [68.23, 80.67, 86.27, 89.27, 93.71],
        "ViM": [55.77, 71.61, 80.05, 82.09, 86.32],
        "GraphMamba": [61.36, 78.07, 87.21, 91.30, 96.58],
        r"S$^2$Mamba": [52.73, 75.50, 86.62, 89.28, 95.04],
        "HCI-Net": [62.87, 80.13, 87.35, 88.35, 95.55],
        "MF3DHeat": [65.94, 83.17, 90.45, 92.12, 97.56],
    },
    "Augsburg": {
        "2D-CNN": [73.11, 77.64, 79.81, 83.12, 84.44],
        "3D-CNN": [74.65, 82.14, 85.29, 85.63, 87.28],
        "HybridSN": [64.52, 73.40, 77.31, 85.21, 86.22],
        "ViT": [77.53, 78.71, 78.64, 82.65, 86.12],
        "MorphFormer": [83.29, 86.01, 87.18, 89.71, 90.24],
        "SSFTT": [84.38, 87.66, 89.21, 90.72, 91.53],
        "ViM": [83.50, 84.43, 84.80, 87.58, 87.78],
        "GraphMamba": [84.49, 86.22, 88.63, 90.83, 91.08],
        r"S$^2$Mamba": [83.32, 87.47, 88.11, 90.79, 91.23],
        "HCI-Net": [84.14, 87.24, 87.82, 88.70, 90.38],
        "MF3DHeat": [84.53, 87.79, 89.33, 90.88, 91.77],
    },
    "Houston": {
        "2D-CNN": [68.18, 80.41, 84.60, 87.58, 88.78],
        "3D-CNN": [68.54, 75.84, 81.48, 86.42, 88.24],
        "HybridSN": [71.52, 77.68, 82.10, 85.99, 88.14],
        "ViT": [67.79, 79.89, 83.38, 87.15, 88.80],
        "MorphFormer": [74.72, 81.94, 88.81, 90.86, 93.29],
        "SSFTT": [62.37, 79.41, 84.59, 88.18, 90.39],
        "ViM": [72.61, 81.82, 83.76, 86.15, 88.93],
        "GraphMamba": [58.69, 82.80, 84.67, 86.87, 90.36],
        r"S$^2$Mamba": [75.11, 82.97, 88.44, 91.43, 93.87],
        "HCI-Net": [70.78, 82.74, 88.82, 90.93, 92.70],
        "MF3DHeat": [76.15, 84.80, 88.97, 91.69, 95.30],
    },
}


METHODS = [
    "2D-CNN",
    "3D-CNN",
    "HybridSN",
    "ViT",
    "MorphFormer",
    "SSFTT",
    "ViM",
    "GraphMamba",
    r"S$^2$Mamba",
    "HCI-Net",
    "MF3DHeat",
]

TRAINING_RATIOS = [20, 40, 60, 80, 100]


# ============================================================
# 2. Figure configuration
# ============================================================

DATASET_ORDER = ["Indian", "Augsburg", "Houston"]

PANEL_LABELS = {
    "Indian": "(a)",
    "Augsburg": "(b)",
    "Houston": "(c)",
}

Y_AXIS_CFG = {
    "Indian": (40, 100),
    "Augsburg": (60, 95),
    "Houston": (55, 100),
}

Y_TICKS = {
    "Indian": [40, 50, 60, 70, 80, 90, 100],
    "Augsburg": [60, 70, 80, 90, 95],
    "Houston": [55, 65, 75, 85, 95, 100],
}

MAIN_METHOD = "MF3DHeat"

STRONG_BASELINES = [
    "MorphFormer",
    "SSFTT",
    "GraphMamba",
    r"S$^2$Mamba",
    "HCI-Net",
]

WEAK_BASELINES = [
    "2D-CNN",
    "3D-CNN",
    "HybridSN",
    "ViT",
    "ViM",
]

MARKERS = {
    "2D-CNN": "o",
    "3D-CNN": "s",
    "HybridSN": "D",
    "ViT": "^",
    "MorphFormer": "v",
    "SSFTT": ">",
    "ViM": "<",
    "GraphMamba": "p",
    r"S$^2$Mamba": "h",
    "HCI-Net": "X",
    "MF3DHeat": "*",
}

STRONG_COLORS = {
    "MorphFormer": "#7B61FF",
    "SSFTT": "#8C564B",
    "GraphMamba": "#7F7F7F",
    r"S$^2$Mamba": "#BCBD22",
    "HCI-Net": "#17BECF",
}

WEAK_COLORS = {
    "2D-CNN": "#9E9E9E",
    "3D-CNN": "#B0B0B0",
    "HybridSN": "#8FA6B2",
    "ViT": "#A8A8A8",
    "ViM": "#B8A9C9",
}

MFH_COLOR = "#B00020"
GRID_COLOR = "#9A9A9A"

OUT_DIR = os.path.join(".", "results", "vis_results")
OUT_NAME = "OA_training_ratio_tripanel_v2"
EXPORT_FORMATS: List[str] = ["png", "pdf"]
PNG_DPI = 600


# ============================================================
# 3. Style utilities
# ============================================================

def choose_available_font() -> str:
    """
    避免 Times New Roman 不存在时出现：
    findfont: Font family 'Times New Roman' not found.

    如果系统中存在 Times New Roman，则使用它；
    否则使用 matplotlib 默认自带的 DejaVu Serif。
    """
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in available_fonts:
        return "Times New Roman"

    if "Times" in available_fonts:
        return "Times"

    return "DejaVu Serif"


def set_global_style() -> None:
    font_name = choose_available_font()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"

    print(f"Using font: {font_name}")


def get_method_style(method: str) -> Dict:
    if method == MAIN_METHOD:
        return dict(
            color=MFH_COLOR,
            linestyle="-",
            marker=MARKERS[method],
            linewidth=2.8,
            markersize=9.5,
            markeredgewidth=0.7,
            markeredgecolor="#650010",
            alpha=1.0,
            zorder=10,
        )

    if method in STRONG_BASELINES:
        return dict(
            color=STRONG_COLORS[method],
            linestyle="-",
            marker=MARKERS[method],
            linewidth=1.5,
            markersize=5.2,
            markeredgewidth=0.4,
            alpha=0.92,
            zorder=5,
        )

    if method in WEAK_BASELINES:
        return dict(
            color=WEAK_COLORS[method],
            linestyle="-",
            marker=MARKERS[method],
            linewidth=1.0,
            markersize=4.6,
            markeredgewidth=0.3,
            alpha=0.55,
            zorder=2,
        )

    return dict(
        color="#999999",
        linestyle="-",
        marker="o",
        linewidth=1.0,
        markersize=4.5,
        alpha=0.55,
        zorder=1,
    )


# ============================================================
# 4. Plot function
# ============================================================

def plot_tripanel_oa() -> List[str]:
    set_global_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.4),
        dpi=300,
        facecolor="white",
    )

    for ax, dataset_name in zip(axes, DATASET_ORDER):
        data = ALL_DATA[dataset_name]

        ax.set_facecolor("white")
        ax.set_axisbelow(True)

        line_handles = []

        for method in METHODS:
            if method not in data:
                raise KeyError(f"{dataset_name} 缺少方法数据: {method}")

            y = data[method]

            if len(y) != len(TRAINING_RATIOS):
                raise ValueError(
                    f"{dataset_name}/{method} 的数据长度应为 {len(TRAINING_RATIOS)}，"
                    f"但得到 {len(y)}"
                )

            style = get_method_style(method)

            line, = ax.plot(
                TRAINING_RATIOS,
                y,
                label=method,
                **style,
            )

            line_handles.append(line)

        # ----------------------------------------------------
        # Axis limits and ticks
        # ----------------------------------------------------
        y_min, y_max = Y_AXIS_CFG[dataset_name]
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(Y_TICKS[dataset_name])

        ax.set_xlim(18, 102)
        ax.set_xticks(TRAINING_RATIOS)

        # ----------------------------------------------------
        # Each subplot has its own x/y labels
        # ----------------------------------------------------
        ax.set_xlabel(
            "Training ratio (%)",
            fontsize=18,
            labelpad=7,
        )

        ax.set_ylabel(
            "Overall Accuracy (%)",
            fontsize=18,
            labelpad=7,
        )

        # ----------------------------------------------------
        # Panel label under each subplot: (a), (b), (c)
        # ----------------------------------------------------
        ax.text(
            0.5,
            -0.22,
            PANEL_LABELS[dataset_name],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="normal",
        )

        # ----------------------------------------------------
        # Weak grid: only y-axis
        # ----------------------------------------------------
        ax.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
            color=GRID_COLOR,
        )
        ax.grid(False, axis="x")

        # ----------------------------------------------------
        # Clean spines
        # ----------------------------------------------------
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=13.5,
            length=4.0,
            width=0.9,
            direction="out",
        )

        # ----------------------------------------------------
        # Legend inside each subplot
        # ----------------------------------------------------
        ax.legend(
            handles=line_handles,
            labels=METHODS,
            loc="lower right",
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#CFCFCF",
            framealpha=0.78,
            fontsize=9.0,
            handlelength=2.0,
            handletextpad=0.45,
            columnspacing=0.80,
            labelspacing=0.35,
            borderpad=0.45,
            markerscale=1.00,
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.965,
        bottom=0.255,
        wspace=0.28,
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    saved_paths = []

    for ext in EXPORT_FORMATS:
        out_path = os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}")

        save_kwargs = dict(
            bbox_inches="tight",
            facecolor="white",
        )

        if ext == "png":
            save_kwargs["dpi"] = PNG_DPI

        fig.savefig(out_path, **save_kwargs)
        saved_paths.append(out_path)
        print(f"Saved figure to: {out_path}")

    plt.close(fig)

    return saved_paths


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    plot_tripanel_oa()


if __name__ == "__main__":
    main()
