import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_image(path, img):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def lighten(A, B):
    return np.maximum(A, B)


def difference(A, B):
    return np.abs(A - B)


def hard_light(A, B):
    C = np.zeros_like(A)

    mask = A <= 0.5
    C[mask] = 2 * A[mask] * B[mask]
    C[~mask] = 1 - 2 * (1 - A[~mask]) * (1 - B[~mask])

    return C


def soft_light(A, B):
    C = np.zeros_like(A)

    mask = A <= 0.5
    C[mask] = (2 * A[mask] - 1) * (B[mask] - B[mask]**2) + B[mask]
    C[~mask] = (2 * A[~mask] - 1) * (np.sqrt(B[~mask]) - B[~mask]) + B[~mask]

    return C



if __name__ == "__main__":
    imgA = load_image("C:\\Users\\rithw\\Downloads\\imageA.png")
    imgB = load_image("C:\\Users\\rithw\\Downloads\\imageB.png")

    assert imgA.shape == imgB.shape, "Images must be the same size"

    out_lighten = lighten(imgA, imgB)
    out_difference = difference(imgA, imgB)
    out_hard_light = hard_light(imgA, imgB)
    out_soft_light = soft_light(imgA, imgB)

    save_image("lighten.png", out_lighten)
    save_image("difference.png", out_difference)
    save_image("hard_light.png", out_hard_light)
    save_image("soft_light.png", out_soft_light)

    print("Blending completed successfully.")

