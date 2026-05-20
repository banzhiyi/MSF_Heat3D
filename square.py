import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def build_grid_image(output_path: str) -> None:
    # Cell addresses are (row, col), 1-based, with row 1 at the top.
    green_cells = {
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (3, 3),
    }
    white_cells = {
        (1, 3),
        (2, 3),
    }

    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")

    for row in range(1, 4):
        for col in range(1, 4):
            if (row, col) in white_cells:
                facecolor = "#FFFFFF"
            elif (row, col) in green_cells:
                facecolor = "#385723"
            else:
                facecolor = "#FFFFFF"

            # Convert top-origin row indexing to matplotlib bottom-origin y.
            y = 3 - row
            x = col - 1
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="#000000",
                    linewidth=0.8,
                )
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    out_file = "results/vis_results/grid_3x3.png"
    build_grid_image(out_file)
    print(f"Saved: {out_file}")
