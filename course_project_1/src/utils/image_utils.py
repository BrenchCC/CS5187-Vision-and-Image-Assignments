"""
Image utility functions for loading and preprocessing

This module provides functions for loading, preprocessing, and creating test images.
"""

import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image

    Parameters:
        image_path (str): Path to the image file

    Returns:
        image_gray (np.ndarray): Grayscale image
    """
    image = Image.open(image_path)
    image_gray = np.array(image.convert('L'))
    return image_gray


def create_test_image_with_corners(size=200):
    """
    Create a test image with corners

    Parameters:
        size (int): Image size

    Returns:
        image (np.ndarray): Test image
    """
    image = np.zeros((size, size), dtype=np.uint8)

    # Draw rectangles to create corners
    # Rectangle 1: (20, 20) to (80, 80)
    image[20:81, 20:81] = 200

    # Rectangle 2: (100, 30) to (150, 90)
    image[30:91, 100:151] = 180

    # Rectangle 3: (50, 120) to (120, 170)
    image[120:171, 50:121] = 160

    return image


def create_shifted_test_image(image, dx=5, dy=5):
    """
    Create a shifted version of the test image

    Parameters:
        image (np.ndarray): Original image
        dx, dy (int): Shift amounts

    Returns:
        shifted (np.ndarray): Shifted image
    """
    h, w = image.shape
    shifted = np.zeros_like(image)

    # Compute source and destination regions
    y_src = max(0, -dy)
    y_dst = max(0, dy)
    x_src = max(0, -dx)
    x_dst = max(0, dx)

    height = min(h - y_dst, h - y_src)
    width = min(w - x_dst, w - x_src)

    shifted[y_dst:y_dst+height, x_dst:x_dst+width] = \
        image[y_src:y_src+height, x_src:x_src+width]

    return shifted
