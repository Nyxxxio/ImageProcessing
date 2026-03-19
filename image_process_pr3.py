import cv2
import numpy as np

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

def uniform_noise(img):
    noise = np.random.uniform(-30, 30, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def jpeg_artifacts(img):
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    return cv2.imdecode(enc, 1)

def mean_filter(img):
    return cv2.blur(img, (5,5))

def median_filter(img):
    return cv2.medianBlur(img, 5)

def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5,5), 0)

noisy_images = [
    ("img1_uniform", uniform_noise(img1)),
    ("img1_jpeg", jpeg_artifacts(img1)),
    ("img2_uniform", uniform_noise(img2)),
    ("img2_jpeg", jpeg_artifacts(img2))
]

for name, img in noisy_images:
    cv2.imwrite(name + ".jpg", img)

for name, img in noisy_images:
    cv2.imwrite(name + "_mean.jpg", mean_filter(img))
    cv2.imwrite(name + "_median.jpg", median_filter(img))
    cv2.imwrite(name + "_gaussian.jpg", gaussian_filter(img))

print("Done")
