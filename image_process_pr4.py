import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread("C:\\Users\\rithw\\uniiiiiiiiiii\\Image Proc\\HW4\\image1.jpg")

image3 = cv2.imread("C:\\Users\\rithw\\uniiiiiiiiiii\\Image Proc\\HW4\\image2.jpg")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

gray2 = add_gaussian_noise(gray1)


def apply_canny(img):
    return cv2.Canny(img, 100, 200)

def sobel_manual(img):
    img = img.astype(np.float32)

    Gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])

    Gy_kernel = np.array([[-1, -2, -1],
                          [0,  0,  0],
                          [1,  2,  1]])

    rows, cols = img.shape
    Gx = np.zeros_like(img)
    Gy = np.zeros_like(img)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = img[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(region * Gx_kernel)
            Gy[i, j] = np.sum(region * Gy_kernel)

    G = np.sqrt(Gx**2 + Gy**2)

    G = (G / G.max()) * 255
    return G.astype(np.uint8)


canny1 = apply_canny(gray1)
sobel1 = sobel_manual(gray1)

canny2 = apply_canny(gray2)
sobel2 = sobel_manual(gray2)

canny3 = apply_canny(gray3)
sobel3 = sobel_manual(gray3)


titles = [
    "Original (Image 1)", "Canny", "Sobel",
    "Noisy (Image 2)", "Canny", "Sobel",
    "Free Image (Image 3)", "Canny", "Sobel"
]

images = [
    gray1, canny1, sobel1,
    gray2, canny2, sobel2,
    gray3, canny3, sobel3
]

plt.figure(figsize=(12, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()


cv2.imwrite("canny_image1.jpg", canny1)
cv2.imwrite("sobel_image1.jpg", sobel1)

cv2.imwrite("canny_image2.jpg", canny2)
cv2.imwrite("sobel_image2.jpg", sobel2)

cv2.imwrite("canny_image3.jpg", canny3)
cv2.imwrite("sobel_image3.jpg", sobel3)
