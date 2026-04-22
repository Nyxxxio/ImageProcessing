import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image_path = r"C:\Users\rithw\uniiiiiiiiiii\Image Proc\HW5\image.jpg"
img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

block_size = 15
C = 5  

adaptive_thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    block_size,
    C
)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

K = 3

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

_, labels, centers = cv2.kmeans(
    pixel_values,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)

segmented_image = centers[labels.flatten()]

segmented_image = segmented_image.reshape(img_rgb.shape)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title("Mean Adaptive Threshold")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.title(f"K-Means (K={K})")
plt.axis("off")

plt.tight_layout()
plt.show()
