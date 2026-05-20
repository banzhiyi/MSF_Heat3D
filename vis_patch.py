import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager


def choose_available_font():
    """
    如果存在 Times New Roman，则优先使用；
    否则使用 matplotlib 自带的 DejaVu Serif。
    """
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    if "Times New Roman" in available_fonts:
        return "Times New Roman"
    elif "Times" in available_fonts:
        return "Times"
    else:
        return "DejaVu Serif"


def set_global_style():
    font_name = choose_available_font()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    print(f"Using font: {font_name}")


def add_subplot_legend(ax):
    """
    每个子图内部单独放置 legend：
    - 右下角；
    - 横向排列；
    - 3 个数据集放在一行。
    """
    ax.legend(
        loc="lower right",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.78,
        fontsize=10.5,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=0.75,
        borderpad=0.35,
        labelspacing=0.25,
        markerscale=0.95,
    )


def plot_hyperparameter_sensitivity():
    set_global_style()

    # ============================================================
    # 1. Experimental data
    # ============================================================

    # ---------------- Patch size ----------------
    patch_sizes = [3, 5, 7, 9, 11]

    patch_data = {
        "Indian Pines": [90.24, 96.26, 97.56, 97.37, 97.28],
        "Augsburg": [88.00, 90.50, 91.77, 91.49, 90.91],
        "Houston 2013": [90.47, 92.94, 94.27, 95.30, 95.06],
    }

    # ---------------- Reduced bands ----------------
    reduced_bands = [12, 24, 36, 48]

    reduced_data = {
        "Indian Pines": [97.10, 97.56, 97.68, 98.16],
        "Augsburg": [91.64, 91.77, 92.38, 91.25],
        "Houston 2013": [93.98, 95.30, 94.43, 94.61],
    }

    # ============================================================
    # 2. Style configuration
    # ============================================================

    dataset_styles = {
        "Indian Pines": {
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
        },
        "Augsburg": {
            "color": "#ff7f0e",
            "linestyle": "-",
            "marker": "s",
        },
        "Houston 2013": {
            "color": "#2ca02c",
            "linestyle": "-",
            "marker": "^",
        },
    }

    # 两个子图统一 y 轴范围
    y_min, y_max = 88, 99
    y_ticks = [88, 90, 92, 94, 96, 98]

    # ============================================================
    # 3. Create figure
    # ============================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.2),
        dpi=300,
        facecolor="white",
    )

    # ============================================================
    # 4. Left subplot: patch size
    # ============================================================

    ax = axes[0]

    for dataset_name, oa_values in patch_data.items():
        style = dataset_styles[dataset_name]

        ax.plot(
            patch_sizes,
            oa_values,
            label=dataset_name,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6.2,
            markeredgewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("Input patch size", fontsize=15)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=15)

    ax.set_xlim(2.6, 11.4)
    ax.set_xticks(patch_sizes)

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.65,
        alpha=0.38,
        color="#8C8C8C",
    )
    ax.grid(False, axis="x")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12.5,
        length=3.8,
        width=0.8,
        direction="out",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.5,
        -0.22,
        "(a)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="normal",
    )

    # 每个子图内部单独放置 legend
    add_subplot_legend(ax)

    # ============================================================
    # 5. Right subplot: reduced bands
    # ============================================================

    ax = axes[1]

    for dataset_name, oa_values in reduced_data.items():
        style = dataset_styles[dataset_name]

        ax.plot(
            reduced_bands,
            oa_values,
            label=dataset_name,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=6.2,
            markeredgewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("Reduced bands", fontsize=15)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=15)

    ax.set_xlim(10, 50)
    ax.set_xticks(reduced_bands)

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.65,
        alpha=0.38,
        color="#8C8C8C",
    )
    ax.grid(False, axis="x")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12.5,
        length=3.8,
        width=0.8,
        direction="out",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.5,
        -0.22,
        "(b)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="normal",
    )

    # 每个子图内部单独放置 legend
    add_subplot_legend(ax)

    # ============================================================
    # 6. Layout and save
    # ============================================================

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.96,
        bottom=0.24,
        wspace=0.28,
    )

    out_dir = "results/vis_results"
    os.makedirs(out_dir, exist_ok=True)

    out_pdf = os.path.join(out_dir, "hyperparameter_sensitivity.pdf")
    out_png = os.path.join(out_dir, "hyperparameter_sensitivity.png")

    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"Saved PDF to: {out_pdf}")
    print(f"Saved PNG to: {out_png}")


def main():
    plot_hyperparameter_sensitivity()


if __name__ == "__main__":
    main()
