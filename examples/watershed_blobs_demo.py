#!/usr/bin/env python3
"""
Watershed Segmentation Demo on blobs-3.tif
------------------------------------------
Assumes 'blobs-3.tif' is in the SAME directory you run this script from.

Pipeline:
1) Load image, convert to grayscale if needed
2) Threshold to binary (Otsu if available; fallback heuristic otherwise)
3) Compute distance transform on the binary foreground
4) Find marker seeds from distance peaks
5) Run watershed to split touching blobs
6) Show figures: original, distance, markers, labels, and boundaries overlay

Usage:
    python watershed_blobs_demo.py
    python watershed_blobs_demo.py --savefig out        # saves out_*.png
    python watershed_blobs_demo.py --minobj 64          # remove tiny objects
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, util, morphology, measure, segmentation, feature
try:
    from skimage.filters import threshold_otsu
    HAVE_OTSU = True
except Exception:
    HAVE_OTSU = False

from scipy import ndimage as ndi


def load_and_binarize(local_name="blobs-3.tif", minobj=0):
    path = Path.cwd() / local_name
    if not path.exists():
        raise FileNotFoundError(f"Could not find '{local_name}' in {Path.cwd()}")

    img = io.imread(str(path))

    if img.ndim == 3:
        img = color.rgb2gray(img)

    if img.dtype == bool:
        binary = img
    else:
        img = util.img_as_float(img)
        thr = None
        if HAVE_OTSU:
            try:
                thr = threshold_otsu(img)
            except Exception:
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
            thr = 0.5*(c1 + c2)
        binary = img > thr

    if minobj and minobj > 0:
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=int(minobj))

    return binary


def marker_seeds_from_distance(binary):
    distance = ndi.distance_transform_edt(binary)

    try:
        maxima_mask = segmentation.local_maxima(distance)
    except Exception:
        coords = feature.peak_local_max(distance, labels=binary, footprint=np.ones((3,3)), exclude_border=False)
        maxima_mask = np.zeros_like(distance, dtype=bool)
        if coords.size:
            maxima_mask[tuple(coords.T)] = True

    maxima_mask &= binary
    markers, _ = ndi.label(maxima_mask)

    if markers.max() < 2:
        dist_smooth = ndi.gaussian_filter(distance, sigma=1.0)
        try:
            maxima_mask = segmentation.local_maxima(dist_smooth) & binary
        except Exception:
            coords = feature.peak_local_max(dist_smooth, labels=binary, footprint=np.ones((3,3)), exclude_border=False)
            maxima_mask = np.zeros_like(distance, dtype=bool)
            if coords.size:
                maxima_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(maxima_mask)

    return distance, markers


def run_watershed(binary):
    distance, markers = marker_seeds_from_distance(binary)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return distance, markers, labels


def show_results(binary, distance, markers, labels, save_prefix=None):
    fig1, axes1 = plt.subplots(1, 2, figsize=(12,5))
    ax = axes1[0]
    ax.imshow(binary, cmap='gray'); ax.set_title("Binary foreground"); ax.axis('off')
    ax = axes1[1]
    im = ax.imshow(distance, cmap='magma'); ax.set_title("Distance transform"); ax.axis('off')
    fig1.colorbar(im, ax=axes1[1], fraction=0.046, pad=0.04)
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12,5))
    ax = axes2[0]
    ax.imshow(distance, cmap='magma')
    ax.contour(markers > 0, colors='cyan', linewidths=1)
    ax.set_title(f"Marker seeds (count={markers.max()})"); ax.axis('off')
    ax = axes2[1]
    ax.imshow(labels, cmap='nipy_spectral', vmin=0)
    ax.set_title(f"Watershed labels (regions={labels.max()})"); ax.axis('off')
    fig2.tight_layout()

    boundaries = segmentation.find_boundaries(labels, mode='outer')
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.imshow(binary, cmap='gray')
    ax3.contour(boundaries, colors='lime', linewidths=1)
    ax3.set_title("Watershed boundaries over binary"); ax3.axis('off')
    fig3.tight_layout()

    if save_prefix:
        fig1.savefig(save_prefix + "_binary_distance.png", dpi=200, bbox_inches="tight")
        fig2.savefig(save_prefix + "_markers_labels.png", dpi=200, bbox_inches="tight")
        fig3.savefig(save_prefix + "_boundaries_overlay.png", dpi=200, bbox_inches="tight")
        print("Saved:")
        print(" ", save_prefix + "_binary_distance.png")
        print(" ", save_prefix + "_markers_labels.png")
        print(" ", save_prefix + "_boundaries_overlay.png")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Watershed demo on blobs-3.tif")
    parser.add_argument("--savefig", type=str, default=None, help="If set, save figures with this prefix.")
    parser.add_argument("--minobj", type=int, default=64, help="Remove objects smaller than this (pixels). Use 0 to disable.")
    args = parser.parse_args()

    binary = load_and_binarize("blobs-3.tif", minobj=args.minobj).astype(bool)
    distance, markers, labels = run_watershed(binary)
    show_results(binary, distance, markers, labels, save_prefix=args.savefig)


if __name__ == "__main__":
    main()
