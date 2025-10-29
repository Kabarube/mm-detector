'''
 # @ Author: Kai Ruben Enerhaugen
 # @ Create Time: 2025-10-24
 # @ Modified by: Kai Ruben Enerhaugen
 # @ Modified time: 2025-10-25
 '''

from skimage import io, filters, transform, util, measure
from skimage.morphology import disk, opening, remove_small_holes, remove_small_objects, label, white_tophat
from skimage.color import rgb2lab, rgb2hsv, rgb2lab, rgb2gray
from skimage.segmentation import *
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries  #TODO Maybe remove this import

from skimage.filters.rank import entropy

from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi   


def load_image(fp="./images/IMG_2754_nonstop_alltogether.JPG", scale_factor=0.2):
  """
  Load an image from disk, apply a fixed crop, resize by a scale factor, and return an 8-bit image.
  Parameters
  ----------
  fp : str, optional
    Filepath to the image to load. Default: "./images/IMG_2754_nonstop_alltogether.JPG".
  scale_factor : float, optional
    Multiplicative factor applied to the original image dimensions to determine the
    output size (height and width). Must be positive. Default: 0.2.
  Returns
  -------
  numpy.ndarray
    The cropped and resized image as a uint8 numpy array with shape
    (new_height, new_width, channels). Resizing is performed with
    skimage.transform.resize using anti_aliasing=True and preserve_range=True.
  Notes
  -----
  - The function performs a fixed crop on the loaded image using util.crop with the
    crop specification ((300, 300), (300, 300), (0, 0)) before resizing.
  - The output dtype is cast to uint8.
  - IO, crop, or resize operations may raise exceptions from skimage or numpy if the
    input file is invalid or the requested operations are not applicable.
  """

  # Load image
  img = io.imread(fp)

  # Resize and crop
  image_cropped = util.crop(img, ((300, 300),(300, 300),(0, 0)))

  height = img.shape[0] * scale_factor
  width = img.shape[1] * scale_factor
  img_resized = transform.resize(image_cropped, (height, width), anti_aliasing=True, preserve_range=True)

  # Return 8bit image
  return img_resized.astype(np.uint8)


"""
We can look at the different channels.
Taking a closer look at the image we can see that we dont have any blue chocolates.
The white paper however, will have some blues
This makes the blue channel a good choice for separating the foreground and background

Alternatively we could have converted the image to HSV or LAB color space
and experiment further, but in this case the blue channel worked very well.


"Rather than standard grayscale conversion, we selected the blue channel from RGB color space.
Analysis revealed that the beige paper background has high blue content while the chocolates
(regardless of their visible color) have lower blue values, providing superior foreground-background separation
compared to luminance-based approaches."
"""

def create_bin_mask(image, scale_factor=0.2):
  img = image
  # Extract channels
  red_c = img[:, :, 0]
  green_c = img[:, :, 1]
  blue_c = img[:, :, 2]

  img_hsv = rgb2hsv(img)
  hue_c = img_hsv[:, :, 0]
  saturation_c = img_hsv[:, :, 1]
  value_c = img_hsv[:, :, 2]

  img_lab = rgb2lab(img)
  light_c = img_lab[:, :, 0]
  green_red_c = img_lab[:, :, 1]
  blue_yellow_c = img_lab[:, :, 2]

  # apply gaussian blur to remove small disturbances
  blue_c = filters.gaussian(blue_c)

  # Create binary mask
  bin_mask = blue_c > filters.threshold_otsu(blue_c, nbins=256)
  bin_mask = remove_small_holes(bin_mask, area_threshold=int(700*scale_factor))
  bin_mask = remove_small_objects(bin_mask, min_size=int(700*scale_factor))
  bin_mask = opening(bin_mask, disk(1))
  bin_mask = ~bin_mask

  return bin_mask


def find_regions(image, mask):
  # 1. Distance transform
  distance = ndi.distance_transform_edt(otsu_mask)

  # 2. Find local maxima
  coords = peak_local_max(distance, min_distance=20, labels=otsu_mask)

  # 3. Create markers and dilate them slightly
  mask = np.zeros(distance.shape, dtype=bool)
  mask[tuple(coords.T)] = True
  markers = label(mask)

  # 4. Apply watershed
  labels = watershed(-distance, markers, mask=otsu_mask)

  ###
  # PLOTTING
  #
  # # Show the segmentation results
  # fig, axes = plt.subplots(2, 2, figsize=(12, 10))

  # axes[0, 0].imshow(otsu_mask, cmap='gray')
  # axes[0, 0].set_title('Binary Mask')

  # axes[0, 1].imshow(distance, cmap='viridis')
  # axes[0, 1].plot(coords[:, 1], coords[:, 0], 'r.', markersize=5)
  # axes[0, 1].set_title('Distance Transform + Markers')

  # axes[1, 0].imshow(label2rgb(labels, bg_label=0))
  # axes[1, 0].set_title('Watershed Labels')

  # axes[1, 1].imshow(img)
  # axes[1, 1].imshow(mark_boundaries(img, labels, color=(0, 0, 1), mode="thick"))
  # axes[1, 1].set_title('Boundaries on Original')

  # plt.tight_layout()
  # plt.show()

  props = measure.regionprops_table(labels, properties=(
    'label',
    "area",
    "perimeter",
    "area_convex",
    'area',
    'centroid',
    'eccentricity',
    'major_axis_length',
    'minor_axis_length',
    'orientation',
    'perimeter'
  ))

  props = measure.regionprops(labels)
  return props, labels #measure.regionprops_table(labels)


img = load_image()
otsu_mask = create_bin_mask(img)

###########
# Measurements and identification
# TODO Create a measure function
#

regions = find_regions(img, otsu_mask)[0]
data = []
for region in regions:
  area = region["area"]
  perimeter = region["perimeter"]
  eccentricity = region["eccentricity"]
  max_ax_length = region["major_axis_length"]
  min_ax_length  = region["minor_axis_length"]
  solidity = region["solidity"]
  circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
  eq_diameter = region["equivalent_diameter_area"]
  extent = region["extent"]
  centroid = region["centroid"]
  
  aspect_ratio = max_ax_length / min_ax_length if min_ax_length > 0 else 0

  # Store measurements
  data.append({
      'label': region["label"],
      'area': area,
      'perimeter': perimeter,
      'circularity': circularity,
      'solidity': solidity,
      'aspect_ratio': aspect_ratio,
      'eccentricity': eccentricity,
      'equivalent_diameter': eq_diameter,
      'extent': extent,
      'centroid_y': centroid[0],
      'centroid_x': centroid[1]
  })


df = pd.DataFrame(data)

print(df)


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Circularity distribution
axes[0, 0].hist(df['circularity'], bins=30, edgecolor='black')
axes[0, 0].set_xlabel('Circularity')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Circularity Distribution')
axes[0, 0].axvline(0.85, color='r', linestyle='--', label='Threshold?')
axes[0, 0].legend()

# Solidity distribution
axes[0, 1].hist(df['solidity'], bins=30, edgecolor='black')
axes[0, 1].set_xlabel('Solidity')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Solidity Distribution')

# Aspect ratio distribution
axes[0, 2].hist(df['aspect_ratio'], bins=30, edgecolor='black')
axes[0, 2].set_xlabel('Aspect Ratio')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Aspect Ratio Distribution')
axes[0, 2].axvline(1.3, color='r', linestyle='--', label='Threshold?')
axes[0, 2].legend()

# Eccentricity distribution
axes[1, 0].hist(df['eccentricity'], bins=30, edgecolor='black')
axes[1, 0].set_xlabel('Eccentricity')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Eccentricity Distribution')

# Area distribution
axes[1, 1].hist(df['area'], bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Area (pixels)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Area Distribution')

# Circularity vs Solidity scatter
axes[1, 2].scatter(df['circularity'], df['solidity'], alpha=0.6)
axes[1, 2].set_xlabel('Circularity')
axes[1, 2].set_ylabel('Solidity')
axes[1, 2].set_title('Circularity vs Solidity')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================
# Classification (Example Thresholds)
# ============================================

# Based on the distributions, classify objects
# These thresholds should be adjusted based on your specific data
# M&Ms are expected to be: highly circular, high solidity, aspect ratio ~1

# Classification criteria for M&Ms:
# - High circularity (> 0.85)
# - Aspect ratio close to 1 (< 1.3)
# - High solidity (> 0.95)



"""
We have decided to exclude the chipped chocolate pieces. Some workarounds could be to use texture to identify the m&m's
regardless of it being chipped or not.
"""

# TODO Create a identify function


# Parameters to identify M&M's
df['is_chipped'] = (
    (df['aspect_ratio'] > 1.3) & 
    (df['solidity'] < 0.97)
)

# Parameters to identify chipped chocolates
median_area = df["area"].median()


circ_th = 0.85
area_th = 3200

df['is_mm'] = (
  #(df["circularity"] < 0.9)

  #(df["aspect_ratio"] > 1.2) 
  (df["eccentricity"] > 0.55) &
  (df["circularity"] < 0.9) &
  (~df["is_chipped"])
   )
  #  
  #  (df["solidity"] > 0.97)

fix, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img)
for idx, row in df.iterrows():
   if row["is_mm"]:
       y, x = row['centroid_y'], row['centroid_x']
       ax[1].plot(x, y, 'ro', markersize=12, markeredgecolor='black', markeredgewidth=1.5)

plt.show()

# print(f"\n=== Classification Results ===")
# print(f"M&Ms detected: {df['is_mm'].sum()}")
# print(f"Other chocolates: {(~df['is_mm']).sum()}")

# # ============================================
# # Visualize Classification Results
# # ============================================

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# labels = find_regions(img, otsu_mask)[1]
# # Original image
# axes[0].imshow(img)
# axes[0].set_title('Original Image')
# axes[0].axis('off')

# # All segmented regions
# axes[1].imshow(label2rgb(labels, image=img, bg_label=0))
# axes[1].set_title(f'All Objects ({len(df)} total)')
# axes[1].axis('off')

# # Classified regions (M&Ms in green, others in red)
# classified_labels = np.zeros_like(labels)
# for idx, row in df.iterrows():
#     if row['is_mm']:
#         classified_labels[labels == row['label']] = 1  # M&Ms
#     else:
#         classified_labels[labels == row['label']] = 2  # Other chocolates

# # Create custom colormap: background=black, M&Ms=green, others=red
# from matplotlib.colors import ListedColormap
# colors = ['black', 'green', 'red']
# cmap = ListedColormap(colors)

# axes[2].imshow(img)
# axes[2].imshow(classified_labels, alpha=0.5, cmap=cmap, vmin=0, vmax=2)
# axes[2].set_title(f'Classification (Green=M&Ms: {df["is_mm"].sum()}, Red=Others: {(~df["is_mm"]).sum()})')
# axes[2].axis('off')

# plt.tight_layout()
# plt.show()






##########
# Save a image with transparent background
#
# # Apply mask to each color channel
# img_masked = img.copy()

# # Remove background
# # Convert to RGBA by adding alpha channel
# img_masked = np.dstack((img, np.ones(img.shape[:2], dtype=np.uint8) * 255))
# # Set background pixels to transparent (alpha = 0)
# img_masked[otsu_mask, 3] = 0

# io.imsave("./images/IMG_2754_nonstop_masked1.png", img_masked)
# # Plot channels
# fig, ax = plt.subplots(3, 3, figsize=(15, 10))

