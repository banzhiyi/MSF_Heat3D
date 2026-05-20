import matplotlib
matplotlib.use("Agg")  # 必须放在 import matplotlib.pyplot as plt 之前
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

# =========================
# Global style
# =========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def interp(p, q, t):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return (1 - t) * p + t * q


def draw_arrow(ax, start, end, lw=1.3, ms=12, color='black', zorder=20):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        zorder=zorder
    )
    ax.add_patch(arrow)


def gaussian_2d(X, Y, cx=0.5, cy=0.5, sigma=0.17):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def draw_front_heatmap(ax, x0, x1, y0, y1, zorder=3):
    """
    Front face:
    continuous 2D heatmap.
    """
    res = 500
    xs = np.linspace(0, 1, res)
    ys = np.linspace(0, 1, res)
    X, Y = np.meshgrid(xs, ys)
    Z = gaussian_2d(X, Y, sigma=0.17)

    front_cmap = LinearSegmentedColormap.from_list(
        "front_heat",
        ["#f6ead0", "#f2d980", "#f3a652", "#e35b3f"]
    )

    ax.imshow(
        Z,
        extent=[x0, x1, y0, y1],
        origin='lower',
        cmap=front_cmap,
        interpolation='bicubic',
        zorder=zorder
    )


def draw_clipped_gradient(ax, polygon_pts, gradient_array, extent, cmap, zorder=1):
    """
    Draw a continuous gradient image and clip it to a polygon.
    This is the key to removing the 'mosaic / strip' artifacts.
    """
    clip_poly = Polygon(
        polygon_pts,
        closed=True,
        facecolor='none',
        edgecolor='none'
    )
    ax.add_patch(clip_poly)

    im = ax.imshow(
        gradient_array,
        extent=extent,
        origin='lower',
        cmap=cmap,
        interpolation='bicubic',
        zorder=zorder
    )
    im.set_clip_path(clip_poly)


def draw_front_grid(ax, x0, x1, y0, y1, n=8, color=(0, 0, 0, 0.22), lw=0.8, zorder=8):
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)

    for xx in xs:
        ax.plot([xx, xx], [y0, y1], color=color, lw=lw, zorder=zorder)
    for yy in ys:
        ax.plot([x0, x1], [yy, yy], color=color, lw=lw, zorder=zorder)


def draw_top_grid(ax, TL, TR, TLB, TRB, n=8, color=(0, 0, 0, 0.18), lw=0.8, zorder=9):
    for t in np.linspace(0, 1, n + 1):
        p0 = interp(TL, TLB, t)
        p1 = interp(TR, TRB, t)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=zorder)

    for t in np.linspace(0, 1, n + 1):
        p0 = interp(TL, TR, t)
        p1 = interp(TLB, TRB, t)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=zorder)


def draw_right_grid(ax, FR, TR, BRB, TRB, n=8, color=(0, 0, 0, 0.18), lw=0.8, zorder=9):
    for t in np.linspace(0, 1, n + 1):
        p0 = interp(FR, BRB, t)
        p1 = interp(TR, TRB, t)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=zorder)

    for t in np.linspace(0, 1, n + 1):
        p0 = interp(FR, TR, t)
        p1 = interp(BRB, TRB, t)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=zorder)


def draw_cube_edges(ax, FL, FR, TR, TL, TLB, TRB, BRB,
                    color=(0, 0, 0, 0.45), lw=1.0, zorder=12):
    # front face
    ax.plot([FL[0], FR[0]], [FL[1], FR[1]], color=color, lw=lw, zorder=zorder)
    ax.plot([FR[0], TR[0]], [FR[1], TR[1]], color=color, lw=lw, zorder=zorder)
    ax.plot([TR[0], TL[0]], [TR[1], TL[1]], color=color, lw=lw, zorder=zorder)
    ax.plot([TL[0], FL[0]], [TL[1], FL[1]], color=color, lw=lw, zorder=zorder)

    # connectors
    ax.plot([TL[0], TLB[0]], [TL[1], TLB[1]], color=color, lw=lw, zorder=zorder)
    ax.plot([TR[0], TRB[0]], [TR[1], TRB[1]], color=color, lw=lw, zorder=zorder)
    ax.plot([FR[0], BRB[0]], [FR[1], BRB[1]], color=color, lw=lw, zorder=zorder)

    # back/top/right edges
    ax.plot([TLB[0], TRB[0]], [TLB[1], TRB[1]], color=color, lw=lw, zorder=zorder)
    ax.plot([TRB[0], BRB[0]], [TRB[1], BRB[1]], color=color, lw=lw, zorder=zorder)


def draw_pseudo_3d_heat_cube(save_path="pseudo_3d_heat_cube.png", dpi=300):
    fig = plt.figure(figsize=(5.8, 4.8), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_aspect("equal")
    ax.axis("off")

    # =========================
    # Cube geometry
    # =========================
    x0, y0 = 0.10, 0.10
    size = 0.56
    dx, dy = 0.18, 0.14

    FL = np.array([x0, y0])
    FR = np.array([x0 + size, y0])
    TR = np.array([x0 + size, y0 + size])
    TL = np.array([x0, y0 + size])

    off = np.array([dx, dy])
    TLB = TL + off
    TRB = TR + off
    BRB = FR + off

    # =========================
    # Top face: continuous clipped gradient
    # =========================
    top_cmap = LinearSegmentedColormap.from_list(
        "top_face",
        ["#e4b96d", "#e7d28b", "#bfd6a2", "#86cae1", "#9a86d2"]
    )
    res = 500
    Xt = np.linspace(0, 1, res)
    Yt = np.linspace(0, 1, res)
    Xtg, Ytg = np.meshgrid(Xt, Yt)
    top_grad = 0.75 * Xtg + 0.25 * (1 - Ytg)

    top_extent = [TL[0], TRB[0], TL[1], TRB[1]]
    draw_clipped_gradient(
        ax,
        [TL, TR, TRB, TLB],
        top_grad,
        top_extent,
        top_cmap,
        zorder=1
    )

    # =========================
    # Right face: continuous clipped gradient
    # =========================
    right_cmap = LinearSegmentedColormap.from_list(
        "right_face",
        ["#9ad1c4", "#6fc4d9", "#6aa3e7", "#8b88d8", "#b08ad5"]
    )
    Xr = np.linspace(0, 1, res)
    Yr = np.linspace(0, 1, res)
    Xrg, Yrg = np.meshgrid(Xr, Yr)
    right_grad = 0.45 * Xrg + 0.55 * Yrg

    right_extent = [FR[0], TRB[0], FR[1], TRB[1]]
    draw_clipped_gradient(
        ax,
        [FR, TR, TRB, BRB],
        right_grad,
        right_extent,
        right_cmap,
        zorder=2
    )

    # =========================
    # Front face heatmap
    # =========================
    draw_front_heatmap(ax, FL[0], FR[0], FL[1], TL[1], zorder=3)

    # =========================
    # Overlay grids
    # =========================
    draw_top_grid(ax, TL, TR, TLB, TRB, n=8, lw=0.75, zorder=9)
    draw_right_grid(ax, FR, TR, BRB, TRB, n=8, lw=0.75, zorder=9)
    draw_front_grid(ax, FL[0], FR[0], FL[1], TL[1], n=8, lw=0.75, zorder=10)

    # =========================
    # Cube edges
    # =========================
    draw_cube_edges(ax, FL, FR, TR, TL, TLB, TRB, BRB, lw=1.0, zorder=12)

    # =========================
    # Heat source
    # =========================
    cx = (FL[0] + FR[0]) / 2
    cy = (FL[1] + TL[1]) / 2

    halo = Circle((cx, cy), 0.055, facecolor="red", edgecolor="none", alpha=0.16, zorder=13)
    core = Circle((cx, cy), 0.016, facecolor="red", edgecolor="darkred", linewidth=1.0, zorder=14)
    ax.add_patch(halo)
    ax.add_patch(core)

    # =========================
    # Front diffusion arrows
    # =========================
    draw_arrow(ax, (cx - 0.08, cy), (cx - 0.24, cy), lw=1.2, ms=11, zorder=20)
    draw_arrow(ax, (cx + 0.08, cy), (cx + 0.24, cy), lw=1.2, ms=11, zorder=20)

    draw_arrow(ax, (cx, cy + 0.08), (cx, cy + 0.24), lw=1.2, ms=11, zorder=20)
    draw_arrow(ax, (cx, cy - 0.08), (cx, cy - 0.24), lw=1.2, ms=11, zorder=20)

    # =========================
    # Depth-direction arrows
    # 新版：一个接近竖直，一个带小夹角
    # =========================
    # shared start point for depth-direction arrows
    depth_start_x = FR[0] - 0.07
    depth_start_y = cy + 0.065

    # Arrow 1: nearly vertical
    draw_arrow(
        ax,
        start=(depth_start_x, depth_start_y),
        end=(depth_start_x, cy + 0.18),
        lw=1.2,
        ms=11,
        zorder=20
    )

    # Arrow 2: 45-degree tilted
    d = 0.075
    draw_arrow(
        ax,
        start=(depth_start_x, depth_start_y),
        end=(depth_start_x + d, depth_start_y + d),
        lw=1.2,
        ms=11,
        zorder=20
    )

    # =========================
    # Limits
    # =========================
    ax.set_xlim(0.04, 0.98)
    ax.set_ylim(0.05, 0.92)

    # =========================
    # Save only
    # =========================
    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
        transparent=False
    )
    plt.close(fig)

    print(f"图片已保存到: {save_path}")


if __name__ == "__main__":
    draw_pseudo_3d_heat_cube("pseudo_3d_heat_cube.png", dpi=300)