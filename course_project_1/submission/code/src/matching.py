"""
Feature Matching Implementation

This module provides feature matching functionality, including:
- Sum of Squared Differences (SSD) matching
- Nearest neighbor search
- Lowe's ratio test
"""

import numpy as np


def match_features(descriptors1, descriptors2, ratio_threshold=0.75):
    """
    Match features using SSD

    Parameters:
        descriptors1 (np.ndarray): Descriptors from image 1 (N, D)
        descriptors2 (np.ndarray): Descriptors from image 2 (M, D)
        ratio_threshold (float): Lowe's ratio threshold

    Returns:
        matches (np.ndarray): Match index pairs (K, 2), each row is (idx1, idx2)
    """
    # Compute distances between all descriptor pairs
    distances = compute_ssd_distances(descriptors1, descriptors2)

    # Find nearest and second nearest neighbors for each descriptor
    matches = find_apply_lowe_ratio(distances, ratio_threshold)

    return matches


def compute_ssd_distances(descriptors1, descriptors2):
    """
    Compute Sum of Squared Differences (SSD) distances between two sets of descriptors

    Parameters:
        descriptors1 (np.ndarray): Descriptors from image 1 (N, D)
        descriptors2 (np.ndarray): Descriptors from image 2 (M, D)

    Returns:
        distances (np.ndarray): Distance matrix (N, M)
    """
    N = descriptors1.shape[0]
    M = descriptors2.shape[0]
    D = descriptors1.shape[1]

    # Use vectorized distance calculation
    # SSD = sum_{k=1}^D (d1k - d2k)^2
    distances = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            diff = descriptors1[i] - descriptors2[j]
            distances[i, j] = np.sum(diff * diff)

    return distances


def find_apply_lowe_ratio(distances, ratio_threshold=0.75):
    """
    Find nearest neighbors and apply Lowe's ratio test

    Parameters:
        distances (np.ndarray): Distance matrix (N, M)
        ratio_threshold (float): Lowe's ratio threshold

    Returns:
        matches (np.ndarray): Match index pairs (K, 2)
    """
    N = distances.shape[0]
    matches = []

    for i in range(N):
        # Find indices of two smallest distances
        sorted_indices = np.argsort(distances[i])
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]

        # Compute ratio
        best_dist = distances[i, best_idx]
        second_best_dist = distances[i, second_best_idx]

        if second_best_dist > 0:
            ratio = best_dist / second_best_dist

            # Apply Lowe's ratio test
            if ratio < ratio_threshold:
                matches.append([i, best_idx])

    return np.array(matches)


def mutual_matching(descriptors1, descriptors2, ratio_threshold=0.75):
    """
    Mutual matching (more strict matching method)

    Parameters:
        descriptors1 (np.ndarray): Descriptors from image 1 (N, D)
        descriptors2 (np.ndarray): Descriptors from image 2 (M, D)
        ratio_threshold (float): Lowe's ratio threshold

    Returns:
        matches (np.ndarray): Mutual match index pairs (K, 2)
    """
    # Forward matching: image1 -> image2
    matches_forward = match_features(descriptors1, descriptors2, ratio_threshold)

    # Backward matching: image2 -> image1
    matches_backward = match_features(descriptors2, descriptors1, ratio_threshold)

    # Keep only mutually consistent matches
    mutual_matches = []
    forward_dict = {m[0]: m[1] for m in matches_forward}

    for j, i in matches_backward:
        if i in forward_dict and forward_dict[i] == j:
            mutual_matches.append([i, j])

    return np.array(mutual_matches)


def compute_correlation_distances(descriptors1, descriptors2):
    """
    Compute normalized cross-correlation distances

    Parameters:
        descriptors1 (np.ndarray): Descriptors from image 1 (N, D)
        descriptors2 (np.ndarray): Descriptors from image 2 (M, D)

    Returns:
        distances (np.ndarray): Distance matrix (N, M)
    """
    N = descriptors1.shape[0]
    M = descriptors2.shape[0]

    distances = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            # Normalize descriptors
            d1 = descriptors1[i]
            d2 = descriptors2[j]
            d1_norm = (d1 - np.mean(d1)) / (np.std(d1) + 1e-8)
            d2_norm = (d2 - np.mean(d2)) / (np.std(d2) + 1e-8)

            # Compute cross-correlation
            correlation = np.sum(d1_norm * d2_norm) / len(d1)
            # Convert to distance (high correlation -> small distance)
            distances[i, j] = 1.0 - correlation

    return distances
