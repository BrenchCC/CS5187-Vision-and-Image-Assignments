"""
RANSAC Outlier Rejection Implementation

This module provides RANSAC algorithm implementation for:
- Robust estimation of geometric transformations
- Outlier filtering
- Homography estimation
- Affine transformation estimation
"""

import numpy as np


def ransac(
    corners1,
    corners2,
    matches,
    num_iterations=1000,
    inlier_threshold=5.0,
    transform_type='homography'
):
    """
    Use RANSAC algorithm to robustly estimate geometric transform and filter outliers

    Parameters:
        corners1 (np.ndarray): Corners from image 1 (N, 2)
        corners2 (np.ndarray): Corners from image 2 (M, 2)
        matches (np.ndarray): Match indices (K, 2)
        num_iterations (int): Number of RANSAC iterations
        inlier_threshold (float): Inlier threshold (pixels)
        transform_type (str): Transform type ('homography' or 'affine')

    Returns:
        best_transform (np.ndarray): Optimal transform matrix
        inlier_matches (np.ndarray): Inlier match indices
        inlier_mask (np.ndarray): Inlier mask
    """
    # Get coordinates of matched points
    points1 = corners1[matches[:, 0]]
    points2 = corners2[matches[:, 1]]

    # Add homogeneous coordinates (y, x) -> (x, y, 1)
    # Note: corners are stored as (y, x), convert to (x, y, 1)
    points1_h = np.column_stack([points1[:, 1], points1[:, 0], np.ones(len(points1))])
    points2_h = np.column_stack([points2[:, 1], points2[:, 0], np.ones(len(points2))])

    best_transform = None
    best_inliers = np.array([])
    best_num_inliers = 0

    # Main RANSAC loop
    for iteration in range(num_iterations):
        # Randomly select minimum sample set
        if transform_type == 'homography':
            sample_indices = np.random.choice(len(matches), 4, replace=False)
        else:  # affine
            sample_indices = np.random.choice(len(matches), 3, replace=False)

        sample1 = points1_h[sample_indices]
        sample2 = points2_h[sample_indices]

        # Estimate transform matrix
        try:
            if transform_type == 'homography':
                transform = estimate_homography(sample1, sample2)
            else:
                transform = estimate_affine(sample1, sample2)

            # Compute reprojection errors for all matches
            errors = compute_reprojection_errors(
                points1_h, points2_h, transform, transform_type
            )

            # Count inliers
            inlier_mask = errors < inlier_threshold**2
            num_inliers = np.sum(inlier_mask)

            # Update best transform
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inlier_mask
                best_transform = transform.copy()

        except np.linalg.LinAlgError:
            continue

    # Re-estimate transform using all inliers
    if best_transform is not None:
        inlier_points1 = points1_h[best_inliers]
        inlier_points2 = points2_h[best_inliers]

        if transform_type == 'homography':
            best_transform = estimate_homography(inlier_points1, inlier_points2)
        else:
            best_transform = estimate_affine(inlier_points1, inlier_points2)

        # Re-compute final errors
        errors = compute_reprojection_errors(
            points1_h, points2_h, best_transform, transform_type
        )
        best_inliers = errors < inlier_threshold**2

    return best_transform, matches[best_inliers], best_inliers


def estimate_homography(points1, points2):
    """
    Estimate homography matrix using DLT (Direct Linear Transform) algorithm

    Parameters:
        points1 (np.ndarray): Source points in homogeneous coordinates (N, 3)
        points2 (np.ndarray): Destination points in homogeneous coordinates (N, 3)

    Returns:
        H (np.ndarray): Homography matrix (3, 3)
    """
    N = len(points1)
    A = np.zeros((2 * N, 9))

    # Build linear system of equations
    for i in range(N):
        x1, y1, w1 = points1[i]
        x2, y2, w2 = points2[i]

        A[2*i] = [0, 0, 0, -w2*x1, -w2*y1, -w2*w1, y2*x1, y2*y1, y2*w1]
        A[2*i + 1] = [w2*x1, w2*y1, w2*w1, 0, 0, 0, -x2*x1, -x2*y1, -x2*w1]

    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize
    H = H / H[2, 2]

    return H


def estimate_affine(points1, points2):
    """
    Estimate affine transform matrix

    Parameters:
        points1 (np.ndarray): Source points in homogeneous coordinates (N, 3)
        points2 (np.ndarray): Destination points in homogeneous coordinates (N, 3)

    Returns:
        A (np.ndarray): Affine transform matrix (3, 3), last row is [0, 0, 1]
    """
    N = len(points1)
    A = np.zeros((2 * N, 6))
    b = np.zeros(2 * N)

    for i in range(N):
        x1, y1, w1 = points1[i]
        x2, y2, w2 = points2[i]

        A[2*i] = [x1, y1, 1, 0, 0, 0]
        A[2*i + 1] = [0, 0, 0, x1, y1, 1]
        b[2*i] = x2
        b[2*i + 1] = y2

    # Least squares solution
    affine_params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Build transform matrix
    affine_matrix = np.eye(3)
    affine_matrix[0, 0] = affine_params[0]
    affine_matrix[0, 1] = affine_params[1]
    affine_matrix[0, 2] = affine_params[2]
    affine_matrix[1, 0] = affine_params[3]
    affine_matrix[1, 1] = affine_params[4]
    affine_matrix[1, 2] = affine_params[5]

    return affine_matrix


def compute_reprojection_errors(points1, points2, transform, transform_type='homography'):
    """
    Compute reprojection errors

    Parameters:
        points1 (np.ndarray): Source points in homogeneous coordinates (N, 3)
        points2 (np.ndarray): Destination points in homogeneous coordinates (N, 3)
        transform (np.ndarray): Transform matrix (3, 3)
        transform_type (str): Transform type

    Returns:
        errors (np.ndarray): Squared errors (N,)
    """
    # Apply transform
    projected_points = apply_transform(points1, transform)

    # Compute squared Euclidean distance
    dx = projected_points[:, 0] - points2[:, 0]
    dy = projected_points[:, 1] - points2[:, 1]
    errors = dx * dx + dy * dy

    return errors


def apply_transform(points, transform):
    """
    Apply transform to a set of points

    Parameters:
        points (np.ndarray): Points in homogeneous coordinates (N, 3)
        transform (np.ndarray): Transform matrix (3, 3)

    Returns:
        transformed_points (np.ndarray): Transformed points (N, 3)
    """
    # Apply transform
    transformed = np.dot(transform, points.T).T

    # Normalize homogeneous coordinates
    mask = transformed[:, 2] != 0
    transformed[mask] = transformed[mask] / transformed[mask, 2:3]

    return transformed


def compute_transform_quality(inlier_mask, num_total_matches):
    """
    Compute transform quality metrics

    Parameters:
        inlier_mask (np.ndarray): Inlier mask
        num_total_matches (int): Total number of matches

    Returns:
        quality (float): Quality score (0-1)
    """
    inlier_ratio = np.sum(inlier_mask) / num_total_matches
    return inlier_ratio
