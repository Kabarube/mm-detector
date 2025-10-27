from skimage.morphology import skeletonize, dilation, opening, footprint_rectangle
from skimage.morphology import erosion, closing
from skimage import io
import matplotlib.pyplot as plt

filename = "images/rhino_detail.tif"

rhino = io.imread(filename)
plt.imshow(rhino, "grey")

# Dilation
rhinodil = dilation(rhino, footprint_rectangle((3, 3)))
plt.imshow(rhinodil, "grey")

# Erosion
rhinoerode = erosion(rhino, footprint_rectangle((3, 3)))
rhinoerode = skeletonize(rhinoerode)
plt.imshow(rhinoerode, "grey")