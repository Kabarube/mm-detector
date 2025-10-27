from skimage import io
from skimage.util import img_as_ubyte, crop
from skimage.transform import resize
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import threshold_otsu, median
from skimage.morphology import disk, binary_closing, binary_opening, opening, dilation, erosion, remove_small_objects
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt

# Adjust image size
img_size = 0.3

# Load, resize and crop image
filepath = "./images/IMG_2754_nonstop_alltogether.JPG"
raw_image = io.imread(filepath)
raw_adj = resize(raw_image, (raw_image.shape[0] * img_size, raw_image.shape[1] * img_size))
raw_adj = crop(raw_adj, crop_width=((100, 100),(100, 100), (0, 0)))

# Grayscale image
gray_img = rgb2gray(raw_adj)

# Hue, Saturation and Value channels
hsv = rgb2hsv(raw_adj)
hue = hsv[:, :, 0]
sat = hsv[:, :, 1]
value = hsv[:, :, 2]


# TODO Cleanup mask for each channel!


# Multichannel Segmentation
# We create a combined mask from the saturation and value channels

# Saturation mask - captures colorful chocolates
# Using Otsu, but we want HIGH saturation (so invert the threshold logic)
sat_threshold = threshold_otsu(sat, nbins=256)
sat_mask = sat > sat_threshold  # High saturation = colorful objects

# Value mask - captures dark chocolates (darker than white background)
# We want LOW value (dark objects), so keep the < comparison
val_threshold = threshold_otsu(value, nbins=256)
val_mask = value < val_threshold  # Low value = dark objects



# # 3. Highlight mask - captures very bright specular reflections ON chocolates
# # Highlights will have HIGH value but are NOT the white background
# # The key: highlights are surrounded by chocolate (they're small bright spots)
# highlight_threshold = 0.9  # Adjust this (0.8-0.95) - very bright pixels
# highlight_mask = value > highlight_threshold

# # Dilate the highlight mask to "fill in" the bright spots
# # This assumes highlights are small and surrounded by chocolate
# selem_highlight = disk(5)  # Adjust size based on highlight size
# highlight_dilated = dilation(highlight_mask, selem_highlight)

# Combine all three masks with OR operation
combined_mask = sat_mask | val_mask

# Use closing with a larger disk to fill holes
selem_fill = disk(6)  # or even 20-30 for your large image
filled_mask = dilation(combined_mask)
filled_mask = binary_closing(combined_mask, selem_fill)
# filled_mask = binary_opening(combined_mask, disk(1))

filled_mask = binary_opening(filled_mask, disk(4))
filled_mask = ~filled_mask
plt.imshow(filled_mask, "gray")

# # Combine masks with OR operation
# combined_mask = sat_mask | val_mask


# selem = disk(3)
# cleaned_mask = binary_opening(combined_mask, selem)
# cleaned_mask = binary_closing(cleaned_mask, selem)
# plt.imshow(cleaned_mask, "gray")




# uint8_image = img_as_ubyte(res_img)


# ###############
# # PREPROCESSING

# # Slight Gaussian blur and contrast adjustment¨
# smooth_img = median(res_img, disk(3))   # Median filter smoothes but preserves edges

# hsv = rgb2hsv(res_img)
# hue_img = hsv[:, :, 0]
# sat_img = hsv[:, :, 1]
# value_img = hsv[:, :, 2]








# binary = uint8_image > threshold_otsu(uint8_image)
# img = find_contours(binary)

# fig, ax = plt.subplots()
# ax.imshow(binary, cmap='gray')

# for contour in img:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

# plt.show()




# from skimage.color import rgb2hsv

# raw_image = io.imread(filepath)
# res_img = resize(raw_image, (raw_image.shape[0] * img_size, raw_image.shape[1] * img_size))
# res_img = crop(res_img, crop_width=((100, 100),(100, 100), (0, 0)))
# hsv = rgb2hsv(res_img)
# hue_img = hsv[:, :, 0]
# sat_img = hsv[:, :, 1]
# value_img = hsv[:, :, 2]

# binary = hue_img > threshold_otsu(hue_img)
# #binary = erosion(binary, disk(3))
# plt.imshow(binary)