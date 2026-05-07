import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

GAUSSIAN_KERNEL = (7, 7)
GAUSSIAN_SIGMA = 1.5

SOBEL_KERNEL = 3

EDGE_THRESHOLD = 80

MORPH_KERNEL_SIZE = 3
MORPH_ITERATIONS = 1

def hybrid_noise_reduction(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=SOBEL_KERNEL
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=SOBEL_KERNEL
    )

    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    _, edge_mask = cv2.threshold(
        gradient_magnitude,
        EDGE_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )
    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        np.uint8
    )

    edge_mask = cv2.dilate(
        edge_mask,
        kernel,
        iterations=MORPH_ITERATIONS
    )

    edge_mask = cv2.morphologyEx(
        edge_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    non_edge_mask = cv2.bitwise_not(edge_mask)

    blurred = cv2.GaussianBlur(
        image,
        GAUSSIAN_KERNEL,
        GAUSSIAN_SIGMA
    )

    edge_part = cv2.bitwise_and(
        image,
        image,
        mask=edge_mask
    )

    smooth_part = cv2.bitwise_and(
        blurred,
        blurred,
        mask=non_edge_mask
    )

    final_result = cv2.add(edge_part, smooth_part)

    return {
        "gray": gray,
        "gradient": gradient_magnitude,
        "edge_mask": edge_mask,
        "blurred": blurred,
        "result": final_result
    }

def show_results(original, outputs, title="Result"):

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(outputs["gray"], cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(outputs["gradient"], cmap="gray")
    plt.title("Sobel Gradient")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(outputs["edge_mask"], cmap="gray")
    plt.title("Edge Mask")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(outputs["blurred"], cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(outputs["result"], cv2.COLOR_BGR2RGB))
    plt.title("Final Hybrid Result")
    plt.axis("off")

    plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()

def add_gaussian_noise(image, mean=0, sigma=30):

    noise = np.random.normal(
        mean,
        sigma,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)

image_path = r"C:\Users\rithw\uniiiiiiiiiii\Image Proc\HW6\Images\image5.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

noisy_image = add_gaussian_noise(image, sigma=35)

outputs = hybrid_noise_reduction(noisy_image)

show_results(noisy_image, outputs, title="Hybrid Noise Reduction")

output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

cv2.imwrite(
    str(output_folder / "final_result.jpg"),
    outputs["result"]
)

print("Processing completed.")
print("Result saved to:", output_folder / "final_result.jpg")
