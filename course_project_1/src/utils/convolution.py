"""
Convolution utilities for image processing

This module provides basic convolution operations used in image processing.
"""

import numpy as np


def convolve2d(image, kernel):
    """
    2D convolution operation

    Parameters:
        image (np.ndarray): Input image (grayscale)
        kernel (np.ndarray): Convolution kernel

    Returns:
        result (np.ndarray): Convolution result
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Boundary padding using reflection
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    result = np.zeros_like(image)

    # Perform convolution
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

    return result
