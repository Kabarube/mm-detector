#!/usr/bin/env python3
"""
Erode Blobs (Local File)
------------------------
Assumes an image file named 'blobs-3.tif' is present in the SAME directory
from which this script is launched. The image can be binary or grayscale (RGB ok);
it will be thresholded to binary if needed.

What it does:
- Loads ./blobs-3.tif
- Converts to grayscale if necessary
- Thresholds to binary (Otsu if available; fallback heuristic if not)
- Applies erosion with different structuring elements
- Shows a 2x4 figure: original + 6 eroded variants (with object counts & area %)

Run:
    python erode_blobs_local.py
Optional:
    python erode_blobs_local.py --savefig out.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, util, measure
from skimage.morphology import erosion, square, disk, diamond, rectangle

# Try to import Otsu; if not available, we fall back to a simple heuristic
try:
    from skimage.filters import threshold_otsu
    HAVE_OTSU = True
except Exception:
    HAVE_OTSU = False


def load_and_binarize(local_name="blobs-3.tif"):
    """
    Load ./blobs-3.tif from the current working directory and return a boolean array.
    """
    path = Path.cwd() / local_name
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{local_name}' in the current working directory: {Path.cwd()}"
        )

    img = io.imread(str(path))

    # Convert to grayscale if needed
    if img.ndim == 3:
        img = color.rgb2gray(img)

    # If already boolean, return
    if img.dtype == bool:
        return img

    # Normalize to [0,1] float
    img = util.img_as_float(img)

    # Threshold to binary
    if HAVE_OTSU:
        try:
            thr = threshold_otsu(img)
        except Exception:
            thr = None
    else:
        thr = None

    if thr is None:
        # Fallback: simple 2-cluster heuristic via percentiles
        flat = img.ravel()
        c1, c2 = np.percentile(flat, 25), np.percentile(flat, 75)
        for _ in range(5):
            d1 = (flat - c1) ** 2
            d2 = (flat - c2) ** 2
            m1 = flat[d1 <= d2].mean() if np.any(d1 <= d2) else c1
            m2 = flat[d2 <  d1].mean() if np.any(d2 <  d1) else c2
            c1, c2 = m1, m2
        thr = 0.5 * (c1 + c2)

    binary = img > thr
    return binary


def analyze(binary: np.ndarray):
    """Return object count and area fraction for a binary image."""
    labeled = measure.label(binary, connectivity=2)
    n = labeled.max()
    area_frac = float(binary.mean())  # fraction of foreground pixels
    return n, area_frac


def erode_with_structuring_elements(binary: np.ndarray):
    """Apply erosion with different structuring elements and return results + labels."""
    ses = [
        ("square(3)", square(3)),
        ("square(5)", square(5)),
        ("disk(3)",   disk(3)),
        ("diamond(3)", diamond(3)),
        ("rectangle(3,5)", rectangle(3, 5)),
        ("cross(3)", np.array([[0,1,0],
                               [1,1,1],
                               [0,1,0]], dtype=bool)),
    ]
    results = []
    for name, se in ses:
        er = erosion(binary, se)
        n, af = analyze(er)
        results.append((name, se, er, n, af))
    return results


def main():
    parser = argparse.ArgumentParser(description="Erode local blobs-3.tif with different structuring elements.")
    parser.add_argument("--savefig", type=str, default=None, help="If set, save the figure to this path.")
    args = parser.parse_args()

    binary = load_and_binarize("blobs-3.tif").astype(bool)

    # Analyze original
    n0, af0 = analyze(binary)
    print(f"Original -> Objects: {n0} | Area: {af0*100:.2f}%")

    # Erode with SEs
    results = erode_with_structuring_elements(binary)

    # Figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()

    # Original
    axes[0].imshow(binary, cmap="gray")
    axes[0].set_title(f"Original\nObjs: {n0} | Area: {af0*100:.1f}%")
    axes[0].axis("off")

    # Show erosions
    for ax, (name, se, eroded, n, af) in zip(axes[1:], results):
        ax.imshow(eroded, cmap="gray")
        ax.set_title(f"Erode: {name}\nObjs: {n} | Area: {af*100:.1f}%")
        ax.axis("off")
        print(f"{name:>14} -> Objects: {n:4d} | Area: {af*100:6.2f}%")

    # Hide any unused axes
    for j in range(1 + len(results), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Erosion of Blobs (./blobs-3.tif) with Different Structuring Elements", y=0.98)
    fig.tight_layout()

    if args.savefig:
        fig.savefig(args.savefig, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.savefig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
