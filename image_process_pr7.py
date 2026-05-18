import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

image_paths = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]

import os
os.makedirs("output", exist_ok=True)

def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def apply_canny(img):
    return cv2.Canny(img, 100, 200)

def sobel_manual(img):
    img = img.astype(np.float32)

    Gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    Gy_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    rows, cols = img.shape
    Gx = np.zeros_like(img)
    Gy = np.zeros_like(img)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = img[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(region * Gx_kernel)
            Gy[i, j] = np.sum(region * Gy_kernel)

    G = np.sqrt(Gx**2 + Gy**2)

    if G.max() != 0:
        G = (G / G.max()) * 255

    return G.astype(np.uint8)

def calculate_metrics(reference, processed):
    mse = mean_squared_error(reference, processed)
    rmse = np.sqrt(mse)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = peak_signal_noise_ratio(reference, processed, data_range=255)

    ssim = structural_similarity(reference, processed, data_range=255)

    return mse, rmse, psnr, ssim

results = []

for idx, path in enumerate(image_paths, start=1):
    image = cv2.imread(path)

    if image is None:
        print(f"Error: Could not load {path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noisy = add_gaussian_noise(gray, sigma=25)

    canny = apply_canny(noisy)
    sobel = sobel_manual(noisy)

    canny_metrics = calculate_metrics(gray, canny)
    sobel_metrics = calculate_metrics(gray, sobel)

    results.append([
        f"Image {idx}", "Canny",
        canny_metrics[0],
        canny_metrics[1],
        canny_metrics[2],
        canny_metrics[3]
    ])

    results.append([
        f"Image {idx}", "Sobel",
        sobel_metrics[0],
        sobel_metrics[1],
        sobel_metrics[2],
        sobel_metrics[3]
    ])

    cv2.imwrite(f"output/image{idx}_original.jpg", gray)
    cv2.imwrite(f"output/image{idx}_noisy.jpg", noisy)
    cv2.imwrite(f"output/image{idx}_canny.jpg", canny)
    cv2.imwrite(f"output/image{idx}_sobel.jpg", sobel)

    titles = [
        "Original",
        "Noisy",
        "Canny",
        "Sobel"
    ]

    images = [
        gray,
        noisy,
        canny,
        sobel
    ]

    plt.figure(figsize=(12, 4))

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.suptitle(f"Image {idx}")
    plt.tight_layout()
    plt.show()

df = pd.DataFrame(
    results,
    columns=[
        "Image",
        "Method",
        "MSE",
        "RMSE",
        "PSNR",
        "SSIM"
    ]
)

df.to_csv("metrics_results.csv", index=False)

print("\nObjective Metric Results:")
print(df.to_string(index=False))

print("\nResults saved to:")
print(" - output/ folder")
print(" - metrics_results.csv")
