'''
 # @ Author: Kai Ruben Enerhaugen
 # @ Create Time: 2025-10-24
 # @ Modified by: Kai Ruben Enerhaugen
 # @ Modified time: 2025-10-29
 '''


from skimage import io, filters, transform, util, measure
from skimage.morphology import disk, opening, remove_small_holes, remove_small_objects, label
from skimage.color import rgb2gray
from skimage.segmentation import watershed
from skimage.filters.rank import entropy
from skimage.feature import peak_local_max
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi   

SCALE_FACTOR = 0.2


t_start = time.time()

def load_image(filepath: str, scale_factor: float = SCALE_FACTOR) -> np.ndarray:

  # Load image
  img = io.imread(filepath)

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
and experiment further, but in this case the blue channel yielded better results.


"Rather than standard grayscale conversion, we selected the blue channel from RGB color space.
Analysis revealed that the beige paper background has high blue content while the chocolates
(regardless of their visible color) have lower blue values, providing superior foreground-background separation
compared to luminance-based approaches."
"""

def create_bin_mask(image: np.ndarray, scale_factor: float = SCALE_FACTOR) -> np.ndarray:

  # Extract channels
  red_c = image[:, :, 0]
  green_c = image[:, :, 1]
  #blue_c = image[:, :, 2]

  # apply gaussian blur on blue channel to remove small disturbances
  blue_c = filters.gaussian(image[:, :, 2])

  # Create binary mask
  bin_mask = blue_c > filters.threshold_otsu(blue_c, nbins=256)
  bin_mask = remove_small_holes(bin_mask, area_threshold=int(700*scale_factor))
  bin_mask = remove_small_objects(bin_mask, min_size=int(700*scale_factor))
  bin_mask = opening(bin_mask, disk(1))
  bin_mask = ~bin_mask

  return bin_mask


def find_regions(image: np.ndarray, binary_mask: np.ndarray) -> pd.DataFrame:
  #Distance transform
  distance = ndi.distance_transform_edt(binary_mask)

  #Find local maxima
  coords = peak_local_max(distance, min_distance=10, labels=binary_mask)

  #Create markers
  markers = np.zeros(distance.shape, dtype=bool)

  markers[tuple(coords.T)] = True
  markers = label(markers)

  # 4. Watershed
  labels = watershed(-distance, markers, mask=binary_mask)

  # 5. Entropy measurement
  gray = rgb2gray(image)
  entropy_img = entropy(gray, disk(5))

  # Extract data
  props = measure.regionprops_table(labels, intensity_image=entropy_img, properties=(
    'label',
    'area',
    'perimeter',
    'intensity_mean',
    'centroid',
    'eccentricity',
    'solidity',
    'major_axis_length',
    'minor_axis_length',
    'orientation',
  ))

  df = pd.DataFrame(props)
  df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2)
  df['aspect_ratio'] = df['major_axis_length'] / df['minor_axis_length']

  return df



def plot_measurements(data:pd.DataFrame):

  fig, ax = plt.subplots(2, 2, figsize=(10, 10))
  

  plots = [
    ('circularity', 'Circularity', 0.9),
    ('solidity', "Solidity", 0.97),
    ('aspect_ratio', 'Aspect Ratio', 1.3),
    ('eccentricity', 'Eccentricity', 0.55),
  ]

  
  for axes, (col, title, threshold) in zip(ax.flat, plots):
    axes.hist(data[col], bins=30, edgecolor='black')
    axes.set_xlabel(title)
    axes.set_ylabel('Count')
    axes.set_title(f"{title} Distribution")
    axes.axvline(threshold, color='r', linestyle='--', label='Threshold')
    axes.legend()


  plt.tight_layout()
  plt.show()


def identify_mm(img:np.ndarray, data:pd.DataFrame):
  # Identification parameteres
  ASPECT_RATIO = 1.3
  SOLIDITY = 0.97
  ECCENTRICITY = 0.55
  CIRCULARITY = 0.9

  """
  We have decided to exclude the chipped chocolate pieces. Some workarounds could be to use texture to identify the m&m's
  regardless of it being chipped or not.
  """
  # Detect chipped chocolates
  data['is_chipped'] = (
    (data['aspect_ratio'] > ASPECT_RATIO) & 
    (data['solidity'] < SOLIDITY))

  # Detect M&M's
  data['is_mm'] = (
    (data["eccentricity"] > ECCENTRICITY) &
    (data["circularity"] < CIRCULARITY) &
    (~data["is_chipped"]))

  # Plot result
  fig, ax = plt.subplots(1, 2, figsize=(12, 6))
  ax[0].imshow(img)
  ax[0].set_title('Original Image')
  ax[1].set_title('Identified')
  

  ax[1].imshow(img)
  for idx, row in data.iterrows():
    if row["is_mm"]:
        y, x = row['centroid-0'], row['centroid-1']
        ax[1].plot(x, y, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=1.5, label="M&M")
        
  ax[1].legend()
  plt.show()



if __name__ == "__main__":
  img = load_image("./images/IMG_2754_nonstop_alltogether.JPG")
  binary_mask = create_bin_mask(img)
  data = find_regions(img, binary_mask)
  plot_measurements(data)
  identify_mm(img, data)


  t_stop = time.time()
  runtime = t_stop - t_start
  print(f"Runtime: {runtime:.3}s")