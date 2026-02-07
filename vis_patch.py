import os

import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，避免 Tk/PIL 工具栏相关报错

import matplotlib.pyplot as plt


def main():
    # X 轴：patch size
    patch_sizes = [3, 5, 7, 9, 11]

    # OA 数据（单位：%）
    oa_indian_pines = [90.24, 96.26, 97.56, 97.37, 97.28]
    oa_augsburg = [88.00, 90.50, 91.77, 91.49, 90.91]
    oa_houston2013 = [90.47, 92.94, 94.27, 95.30, 95.06]

    # 输出目录与文件名
    out_dir = "results/vis_results"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "patch_oa.png")
    out_pdf = os.path.join(out_dir, "patch_oa.pdf")

    # 画布
    plt.figure(figsize=(7.6, 4.6), dpi=160)

    # 不同颜色与样式折线
    plt.plot(
        patch_sizes,
        oa_indian_pines,
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=2.0,
        markersize=5,
        label="IndianPines",
    )
    plt.plot(
        patch_sizes,
        oa_augsburg,
        color="#ff7f0e",
        linestyle="--",
        marker="s",
        linewidth=2.0,
        markersize=5,
        label="Augsburg",
    )
    plt.plot(
        patch_sizes,
        oa_houston2013,
        color="#2ca02c",
        linestyle="-.",
        marker="^",
        linewidth=2.0,
        markersize=5,
        label="Houston2013",
    )

    # 坐标轴与刻度
    plt.xlabel("Input patch size")
    plt.ylabel("OA(%)")
    plt.xticks([3, 5, 7, 9, 11])
    plt.yticks([88, 90, 92, 94, 96, 98])
    plt.xlim(3, 11)
    plt.ylim(88, 98)

    # 网格与图例（右下角）
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()

    # 保存（不调用 show）
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

