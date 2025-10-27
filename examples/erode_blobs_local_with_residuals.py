#!/usr/bin/env python3
"""
Erode Blobs (Local File + Residuals)
------------------------------------
Assumes an image file named 'blobs-3.tif' is present in the SAME directory
from which this script is launched. The image can be binary or grayscale (RGB ok);
it will be thresholded to binary if needed.

What it does:
- Loads ./blobs-3.tif
- Converts to grayscale if necessary
- Thresholds to binary (Otsu if available; fallback heuristic if not)
- Applies erosion with different structuring elements
- Shows two figures:
    (1) Original + eroded variants
    (2) Residuals (original – eroded), i.e. pixels removed
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util, measure
from skimage.morphology import erosion, square, disk, diamond, rectangle

# Otsu threshold if available
try:
    from skimage.filters import threshold_otsu
    HAVE_OTSU = True
except Exception:
    HAVE_OTSU = False


def load_and_binarize(local_name="blobs-3.tif"):
    """Load ./blobs-3.tif and return a boolean array."""
    path = Path.cwd() / local_name
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{local_name}' in the current working directory: {Path.cwd()}"
        )

    img = io.imread(str(path))

    if img.ndim == 3:
        img = color.rgb2gray(img)

    if img.dtype == bool:
        return img

    img = util.img_as_float(img)

    if HAVE_OTSU:
        try:
            thr = threshold_otsu(img)
        except Exception:
            thr = None
    else:
        thr = None

    if thr is None:
        flat = img.ravel()
        c1, c2 = np.percentile(flat, 25), np.percentile(flat, 75)
        for _ in range(5):
            d1 = (flat - c1) ** 2
            d2 = (flat - c2) ** 2
            m1 = flat[d1 <= d2].mean() if np.any(d1 <= d2) else c1
            m2 = flat[d2 <  d1].mean() if np.any(d2 <  d1) else c2
            c1, c2 = m1, m2
        thr = 0.5 * (c1 + c2)

    return img > thr


def analyze(binary: np.ndarray):
    labeled = measure.label(binary, connectivity=2)
    return labeled.max(), float(binary.mean())


def erode_with_structuring_elements(binary: np.ndarray):
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
        res = binary & (~er)  # residual = pixels removed
        results.append((name, se, er, res, n, af))
    return results


def main():
    parser = argparse.ArgumentParser(description="Erode local blobs-3.tif with different structuring elements.")
    parser.add_argument("--savefig", type=str, default=None, help="If set, save the figures to this prefix (adds _eroded/_residuals).")
    args = parser.parse_args()

    binary = load_and_binarize("blobs-3.tif").astype(bool)

    n0, af0 = analyze(binary)
    print(f"Original -> Objects: {n0} | Area: {af0*100:.2f}%")

    results = erode_with_structuring_elements(binary)

    # Figure 1: Original + eroded images
    fig1, axes1 = plt.subplots(2, 4, figsize=(14, 7))
    axes1 = axes1.ravel()
    axes1[0].imshow(binary, cmap="gray")
    axes1[0].set_title(f"Original\nObjs: {n0} | Area: {af0*100:.1f}%")
    axes1[0].axis("off")
    for ax, (name, _, er, _, n, af) in zip(axes1[1:], results):
        ax.imshow(er, cmap="gray")
        ax.set_title(f"Erode: {name}\nObjs: {n} | Area: {af*100:.1f}%")
        ax.axis("off")
        print(f"{name:>14} -> Objects: {n:4d} | Area: {af*100:6.2f}%")
    for j in range(1 + len(results), len(axes1)):
        axes1[j].axis("off")
    fig1.suptitle("Erosion of Blobs with Different Structuring Elements", y=0.98)
    fig1.tight_layout()

    # Figure 2: Residuals (what got removed)
    fig2, axes2 = plt.subplots(2, 4, figsize=(14, 7))
    axes2 = axes2.ravel()
    axes2[0].imshow(binary, cmap="gray")
    axes2[0].set_title("Original")
    axes2[0].axis("off")
    for ax, (name, _, _, res, _, _) in zip(axes2[1:], results):
        ax.imshow(res, cmap="gray")
        ax.set_title(f"Residual: {name}")
        ax.axis("off")
    for j in range(1 + len(results), len(axes2)):
        axes2[j].axis("off")
    fig2.suptitle("Residuals = Original - Eroded", y=0.98)
    fig2.tight_layout()

    if args.savefig:
        fig1.savefig(args.savefig + "_eroded.png", dpi=200, bbox_inches="tight")
        fig2.savefig(args.savefig + "_residuals.png", dpi=200, bbox_inches="tight")
        print(f"Saved figures to {args.savefig}_eroded.png and {args.savefig}_residuals.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
