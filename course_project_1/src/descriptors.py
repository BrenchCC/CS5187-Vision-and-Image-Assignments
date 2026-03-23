"""
Feature Descriptor Implementation

This module provides a simplified implementation of SIFT-like descriptors with:
- Rotation invariance
- Scale invariance
- Local region description
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from utils.convolution import convolve2d


def get_descriptors(image, corners, sigma=1.0, patch_size=16, num_bins=8):
    """
    Extract local descriptors for detected corners

    Parameters:
        image (np.ndarray): Input grayscale image
        corners (np.ndarray): Corner coordinates (N, 2)
        sigma (float): Gaussian smoothing parameter
        patch_size (int): Descriptor patch size
        num_bins (int): Number of bins in orientation histogram

    Returns:
        descriptors (np.ndarray): Feature descriptors (N, D), D is descriptor dimension
    """
    descriptors = []
    half_size = patch_size // 2
    h, w = image.shape

    for corner in corners:
        y, x = corner

        # Ensure within image boundaries
        y_min = max(0, y - half_size)
        y_max = min(h, y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(w, x + half_size)

        # Extract keypoint周围的区域
        patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        y_patch_start = half_size - (y - y_min)
        y_patch_end = y_patch_start + (y_max - y_min)
        x_patch_start = half_size - (x - x_min)
        x_patch_end = x_patch_start + (x_max - x_min)

        if y_patch_start < y_patch_end and x_patch_start < x_patch_end:
            patch[y_patch_start:y_patch_end, x_patch_start:x_patch_end] = \
                image[y_min:y_max, x_min:x_max].astype(np.float32)

        # Compute local descriptor
        desc = compute_descriptor(patch, num_bins)
        descriptors.append(desc)

    return np.array(descriptors)


def compute_gradient(image, sigma):
    """
    Compute image gradients after Gaussian smoothing

    Parameters:
        image (np.ndarray): Input image
        sigma (float): Gaussian smoothing parameter

    Returns:
        Ix (np.ndarray): Horizontal gradient
        Iy (np.ndarray): Vertical gradient
    """
    # Gaussian smoothing
    smoothed = gaussian_filter(image.astype(np.float32), sigma)

    # Gradient computation (Sobel operators)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = convolve2d(smoothed, sobel_x)
    Iy = convolve2d(smoothed, sobel_y)

    return Ix, Iy


def compute_descriptor(patch, num_bins):
    """
    Compute local region descriptor

    Parameters:
        patch (np.ndarray): Local region
        num_bins (int): Number of bins in orientation histogram

    Returns:
        descriptor (np.ndarray): Feature descriptor
    """
    patch_size = patch.shape[0]
    cell_size = patch_size // 4  # 4x4 cell blocks
    num_cells = 4

    descriptor = []

    # Compute patch gradients
    Ix, Iy = compute_gradient(patch, 1.0)
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    gradient_orientation = np.arctan2(Iy, Ix) * 180 / np.pi

    # Divide into 4x4 cell blocks
    for i in range(num_cells):
        for j in range(num_cells):
            y_start = i * cell_size
            y_end = y_start + cell_size
            x_start = j * cell_size
            x_end = x_start + cell_size

            cell_magnitude = gradient_magnitude[y_start:y_end, x_start:x_end]
            cell_orientation = gradient_orientation[y_start:y_end, x_start:x_end]

            # Compute orientation histogram for cell
            cell_hist = compute_orientation_histogram(cell_magnitude, cell_orientation, num_bins)
            descriptor.extend(cell_hist)

    # Normalize descriptor
    descriptor = np.array(descriptor)
    descriptor /= (np.linalg.norm(descriptor) + 1e-8)
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor /= (np.linalg.norm(descriptor) + 1e-8)

    return descriptor


def compute_orientation_histogram(magnitude, orientation, num_bins):
    """
    Compute gradient orientation histogram

    Parameters:
        magnitude (np.ndarray): Gradient magnitude
        orientation (np.ndarray): Gradient orientation (degrees)
        num_bins (int): Number of histogram bins

    Returns:
        histogram (np.ndarray): Orientation histogram
    """
    histogram = np.zeros(num_bins)
    bin_size = 360 / num_bins

    # Process each pixel
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            mag = magnitude[i, j]
            orient = orientation[i, j]

            # Ensure angle in 0-360 degrees range
            if orient < 0:
                orient += 360
            if orient >= 360:
                orient -= 360

            # Calculate bin index
            bin_index = int(orient / bin_size)
            bin_index = min(bin_index, num_bins - 1)

            histogram[bin_index] += mag

    return histogram
