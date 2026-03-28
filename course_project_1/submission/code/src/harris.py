"""
Harris Corner Detector Implementation

This module provides a complete implementation of Harris corner detection, including:
- Image gradient computation
- Structure tensor matrix computation
- Gaussian window application
- Harris response function calculation
- Local maxima detection
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from utils.convolution import convolve2d


def get_harris_corners(image, sigma=1.0, k=0.04, threshold=0.01):
    """
    Detect Harris corners in an image

    Parameters:
        image (np.ndarray): Input grayscale image (H, W)
        sigma (float): Standard deviation for Gaussian smoothing
        k (float): Constant for Harris response function
        threshold (float): Threshold for corner detection

    Returns:
        corners (np.ndarray): Detected corner coordinates (N, 2), where N is the number of corners
        harris_response (np.ndarray): Harris response image (H, W)
    """
    # Step 1: Compute image gradients
    Ix, Iy = compute_gradients(image)

    # Step 2: Compute gradient products
    Ix2 = gaussian_filter(Ix**2, sigma)
    Iy2 = gaussian_filter(Iy**2, sigma)
    Ixy = gaussian_filter(Ix*Iy, sigma)

    # Step 3: Compute Harris response function
    det_M = Ix2 * Iy2 - Ixy**2
    trace_M = Ix2 + Iy2
    harris_response = det_M - k * (trace_M**2)

    # Step 4: Thresholding and local maxima detection
    corners = get_local_maxima(harris_response, threshold)

    return corners, harris_response


def compute_gradients(image):
    """
    Compute horizontal and vertical gradients of an image

    Parameters:
        image (np.ndarray): Input grayscale image

    Returns:
        Ix (np.ndarray): Horizontal gradient
        Iy (np.ndarray): Vertical gradient
    """
    # Sobel operators
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)

    # Compute gradients
    Ix = convolve2d(image.astype(np.float32), sobel_x)
    Iy = convolve2d(image.astype(np.float32), sobel_y)

    return Ix, Iy


def get_local_maxima(harris_response, threshold):
    """
    Find local maxima in the Harris response

    Parameters:
        harris_response (np.ndarray): Harris response image
        threshold (float): Response threshold

    Returns:
        corners (np.ndarray): Local maxima coordinates (N, 2)
    """
    h, w = harris_response.shape
    corners = []

    # Thresholding
    threshold_value = threshold * np.max(harris_response)

    # Local maxima detection (3x3 neighborhood)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            local_region = harris_response[i-1:i+2, j-1:j+2]
            if (harris_response[i, j] > threshold_value and
                harris_response[i, j] == np.max(local_region)):
                corners.append([i, j])

    return np.array(corners)
