#!/usr/bin/env python3
"""
Show Structuring Elements (Kernels)
----------------------------------
Visualize common morphological structuring elements as binary grids.

Kernels included (version-dependent):
- square(3), square(5)
- rectangle(3,5), line: rectangle(1,7) & rectangle(7,1)
- disk(radius=3)
- diamond(radius=3)
- cross(3)  [custom 3x3 cross]
- octagon(3,2)  [if available in your scikit-image]

Usage:
    python show_structuring_elements.py
    python show_structuring_elements.py --savefig se_kernels.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import square, rectangle, disk, diamond
# Try optional elements (octagon) depending on scikit-image version
try:
    from skimage.morphology import octagon  # may not exist in all versions
    HAVE_OCT = True
except Exception:
    HAVE_OCT = False

def make_cross(size=3):
    """Create a simple cross (plus) structuring element of odd size."""
    if size % 2 == 0:
        raise ValueError("Cross size must be odd.")
    se = np.zeros((size, size), dtype=bool)
    mid = size // 2
    se[mid, :] = True
    se[:, mid] = True
    return se

def show_kernel(ax, kernel: np.ndarray, title: str):
    """Render a binary kernel on a square-grid subplot with grid lines."""
    k = kernel.astype(int)
    ax.imshow(k, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    # Grid lines
    h, w = k.shape
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

def main():
    parser = argparse.ArgumentParser(description="Visualize morphological structuring elements (kernels).")
    parser.add_argument("--savefig", type=str, default=None, help="If set, save figure to this path instead of showing.")
    args = parser.parse_args()

    # Build kernels
    kernels = []
    kernels.append(("square(3)", square(3)))
    kernels.append(("square(5)", square(5)))
    kernels.append(("rectangle(3,5)", rectangle(3, 5)))
    kernels.append(("line horiz (1x7)", rectangle(1, 7)))
    kernels.append(("line vert (7x1)", rectangle(7, 1)))
    kernels.append(("disk(r=3)", disk(3)))
    kernels.append(("diamond(r=3)", diamond(3)))
    kernels.append(("cross(3)", make_cross(3)))
    if HAVE_OCT:
        try:
            kernels.append(("octagon(3,2)", octagon(3, 2)))
        except Exception:
            pass  # silently ignore if signature differs

    # Figure layout (up to 3 rows x 3 cols)
    n = len(kernels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, ker) in zip(axes, kernels):
        show_kernel(ax, ker, f"{name}\nshape={ker.shape}")
    # Hide extra axes
    for ax in axes[len(kernels):]:
        ax.axis("off")

    fig.suptitle("Morphological Structuring Elements (Kernels)", y=0.98, fontsize=14)
    fig.tight_layout()

    if args.savefig:
        fig.savefig(args.savefig, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.savefig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
