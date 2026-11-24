"""Whole-image preprocessing with HOG, LBP, and FFT frequency features."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = (256, 256)  # Increased from 128 to preserve fine artifacts
LBP_POINTS = 24
LBP_RADIUS = 3
LBP_BINS = LBP_POINTS + 2
HOG_FEATURE_SIZE = 8100
FFT_FEATURE_SIZE = 64     # Size of frequency spectrum features


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes as a BGR image."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("The uploaded file is not a supported image.")
    if image.shape[0] * image.shape[1] > 40_000_000:
        raise ValueError("The image dimensions exceed the 40-megapixel limit.")
    return image


def prepare_image(image: np.ndarray) -> np.ndarray:
    """Resize and pad the whole image while preserving its aspect ratio."""
    if image is None or image.size == 0:
        raise ValueError("Cannot process an empty image.")

    height, width = image.shape[:2]

    scale = min(IMAGE_SIZE[0] / width, IMAGE_SIZE[1] / height)
    resized_width = min(IMAGE_SIZE[0], max(1, round(width * scale)))
    resized_height = min(IMAGE_SIZE[1], max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    horizontal_padding = IMAGE_SIZE[0] - resized_width
    vertical_padding = IMAGE_SIZE[1] - resized_height
    left = horizontal_padding // 2
    top = vertical_padding // 2
    return cv2.copyMakeBorder(
        resized,
        top,
        vertical_padding - top,
        left,
        horizontal_padding - left,
        cv2.BORDER_REPLICATE,
    )


def extract_fft_features(gray_image: np.ndarray) -> np.ndarray:
    """Extract frequency domain features using 2D Fast Fourier Transform (FFT)."""
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Resize or crop the center spectrum to a fixed length vector
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    half_size = FFT_FEATURE_SIZE // 2
    center_crop = magnitude_spectrum[
        center_h - half_size : center_h + half_size,
        center_w - half_size : center_w + half_size
    ]
    return center_crop.flatten().astype(np.float32)


def features_from_image(image: np.ndarray) -> np.ndarray:
    """Build a fixed-length feature vector combining HOG, LBP, and FFT frequency features."""
    resized = prepare_image(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 1. Spatial shape features (HOG)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),  # Adjusted for 256x256 grid
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    
    # 2. Local texture features (LBP)
    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )
    lbp_histogram, _ = np.histogram(
        lbp.ravel(),
        bins=LBP_BINS,
        range=(0, LBP_BINS),
        density=True,
    )
    
    # 3. Frequency domain fingerprints (FFT)
    fft_features = extract_fft_features(gray)
    
    return np.concatenate((hog_features, lbp_histogram, fft_features)).astype(np.float32)


def texture_hotspots(image: np.ndarray, grid_size: int = 8, limit: int = 10) -> list[dict[str, float]]:
    """Return normalized high-gradient grid locations for a display-only texture map."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    height, width = magnitude.shape
    cell_height = max(1, height // grid_size)
    cell_width = max(1, width // grid_size)
    cells: list[tuple[float, int, int]] = []

    for row in range(grid_size):
        for column in range(grid_size):
            y0, x0 = row * cell_height, column * cell_width
            cell = magnitude[y0 : min(y0 + cell_height, height), x0 : min(x0 + cell_width, width)]
            cells.append((float(cell.mean()), column, row))

    maximum = max((value for value, _, _ in cells), default=0.0)
    if maximum <= 0:
        return []

    strongest = sorted(cells, reverse=True)[:limit]
    return [
        {
            "x": round((column + 0.5) / grid_size, 4),
            "y": round((row + 0.5) / grid_size, 4),
            "intensity": round(value / maximum, 4),
        }
        for value, column, row in strongest
    ]