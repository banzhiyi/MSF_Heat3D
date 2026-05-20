import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# =========================
# Global style
# =========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def draw_2d_heat_conduction(save_path="2d_heat_diffusion.png", dpi=300):
    # 让坐标轴直接铺满整个画布，减少系统默认边距
    fig = plt.figure(figsize=(6, 6), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================
    # Heatmap area
    # 尽量铺满画布，减少白边
    # =========================
    x0, x1 = 0.0, 1.0
    y0, y1 = 0.0, 1.0

    # Generate Gaussian heat diffusion field
    n = 500
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    sigma = 0.35
    Z = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    # Draw heatmap
    ax.imshow(
        Z,
        extent=[x0, x1, y0, y1],
        origin="lower",
        cmap="RdYlBu_r",
        alpha=0.95,
        zorder=1
    )

    # =========================
    # Grid lines
    # =========================
    num_grid = 8
    grid_color = "#9a9a9a"
    xs = np.linspace(x0, x1, num_grid + 1)
    ys = np.linspace(y0, y1, num_grid + 1)

    for xx in xs:
        ax.plot([xx, xx], [y0, y1], color=grid_color, lw=0.8, alpha=0.75, zorder=3)
    for yy in ys:
        ax.plot([x0, x1], [yy, yy], color=grid_color, lw=0.8, alpha=0.75, zorder=3)

    # Grid border
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        color="#7f7f7f",
        lw=1.0,
        zorder=4
    )

    # =========================
    # Center heat source
    # =========================
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    source_radius = 0.015

    heat_source = Circle(
        (cx, cy),
        source_radius,
        facecolor="red",
        edgecolor="darkred",
        lw=1.0,
        zorder=6
    )
    ax.add_patch(heat_source)

    # =========================
    # Diffusion arrows
    # 更长一点，并且从热源外围开始
    # =========================
    directions = [
        (0, 1),    # up
        (0, -1),   # down
        (1, 0),    # right
        (-1, 0),   # left
        (1, 1),    # upper-right
        (-1, 1),   # upper-left
        (1, -1),   # lower-right
        (-1, -1)   # lower-left
    ]

    start_radius = 0.08
    end_radius = 0.27

    for dx, dy in directions:
        norm = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / norm, dy / norm

        start_x = cx + start_radius * ux
        start_y = cy + start_radius * uy
        end_x = cx + end_radius * ux
        end_y = cy + end_radius * uy

        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle='-|>',
            mutation_scale=14,
            linewidth=1.1,
            color='black',
            zorder=7
        )
        ax.add_patch(arrow)

    # =========================
    # Save only
    # =========================
    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white"
    )
    plt.close(fig)

    print(f"图片已保存到: {save_path}")


if __name__ == "__main__":
    draw_2d_heat_conduction("2d_heat_diffusion.png", dpi=300)
