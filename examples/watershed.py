from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage import io
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

"""
Performs watershed segmentation on a grayscale blob image to separate connected objects.
This script demonstrates the watershed segmentation algorithm using the following steps:
1. Loads a grayscale image of blobs
2. Converts it to binary using Otsu thresholding
3. Computes the distance transform
4. Finds local maxima as markers
5. Applies watershed segmentation
6. Visualizes the segmented result
Dependencies:
    - scikit-image
    - scipy
    - matplotlib
    - numpy
The watershed algorithm treats the image as a topographic surface where bright pixels
are high and dark pixels are low. Starting from marker points, it simulates water 
rising from these markers, with watersheds being created where different water 
sources meet.
Parameters:
    filename (str): Path to the input image file
Note:
    The input image should contain bright objects on a dark background.
    The script assumes the image can be loaded as grayscale.
"""


filename = "./images/blobs.tif"
blob = io.imread(filename, as_gray=True)

# Binary convertion using otsu thresholding
blob = blob > threshold_otsu(blob)

distance = ndi.distance_transform_edt(blob)
local_maxi = peak_local_max(distance, footprint=np.ones((3, 3)), labels=blob)

# Create a boolean array with the same shape as distance
peaks_mask = np.zeros_like(distance, dtype=bool)
# Use the local_maximum indices and set them to True
peaks_mask[tuple(local_maxi.T)] = True

# Find the markers (starting points) where the "water" should start flowing
markers = ndi.label(peaks_mask)[0]

# Start watershedding
labels = watershed(-distance, markers, mask=blob)
plt.imshow(labels, "nipy_spectral")


# MEASURE EXAMPLE
# Create a table of properties for the objects
from skimage import measure
properties = measure.regionprops(labels)
for prop in properties:
    print(prop.perimeter, prop.area, prop.centroid, prop.eccentricity)