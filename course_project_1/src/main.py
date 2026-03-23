#!/usr/bin/env python3
"""
Course Project 1: Image Feature Matching and Outlier Rejection

This program demonstrates the complete image feature matching pipeline:
1. Reading and preprocessing input images
2. Harris corner detection
3. Feature descriptor extraction
4. Feature matching
5. RANSAC outlier rejection
6. Result visualization
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    """
    Parse command line arguments

    Returns:
        args (argparse.Namespace): Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Image Feature Matching and Outlier Rejection'
    )

    parser.add_argument(
        'image1',
        help='Path to the first image'
    )

    parser.add_argument(
        'image2',
        help='Path to the second image'
    )

    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='Gaussian smoothing parameter (default: 1.0)'
    )

    parser.add_argument(
        '--k',
        type=float,
        default=0.04,
        help='Harris response function constant (default: 0.04)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='Corner detection threshold (default: 0.01)'
    )

    parser.add_argument(
        '--ratio-threshold',
        type=float,
        default=0.75,
        help='Lowe ratio threshold (default: 0.75)'
    )

    parser.add_argument(
        '--ransac-iterations',
        type=int,
        default=1000,
        help='RANSAC iterations (default: 1000)'
    )

    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=5.0,
        help='RANSAC inlier threshold (pixels, default: 5.0)'
    )

    parser.add_argument(
        '--transform',
        choices=['homography', 'affine'],
        default='homography',
        help='Transform type (homography or affine, default: homography)'
    )

    return parser.parse_args()


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


def visualize_results(image1, image2, corners1, corners2, matches, inlier_matches):
    """
    Visualize matching results

    Parameters:
        image1 (np.ndarray): First image
        image2 (np.ndarray): Second image
        corners1 (np.ndarray): Corners from image1
        corners2 (np.ndarray): Corners from image2
        matches (np.ndarray): All matches
        inlier_matches (np.ndarray): Inlier matches
    """
    # Create combined image
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    combined = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    combined[:h1, :w1] = image1
    combined[:h2, w1:w1+w2] = image2

    plt.figure(figsize=(15, 8))
    plt.imshow(combined, cmap='gray')

    # Draw all matches (red)
    for i, j in matches:
        x1, y1 = corners1[i][1], corners1[i][0]
        x2, y2 = corners2[j][1] + w1, corners2[j][0]
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=0.5, alpha=0.3)

    # Draw inlier matches (green)
    for i, j in inlier_matches:
        x1, y1 = corners1[i][1], corners1[i][0]
        x2, y2 = corners2[j][1] + w1, corners2[j][0]
        plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1, alpha=0.7)

    # Draw corners
    plt.scatter(
        [p[1] for p in corners1],
        [p[0] for p in corners1],
        c='b', s=20, alpha=0.5, label='Image1 corners'
    )
    plt.scatter(
        [p[1] + w1 for p in corners2],
        [p[0] for p in corners2],
        c='orange', s=20, alpha=0.5, label='Image2 corners'
    )

    plt.title(f'Feature Matching Results: {len(matches)} total matches, {len(inlier_matches)} inliers')
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main program
    """
    args = parse_args()

    print("=" * 80)
    print("Image Feature Matching Program")
    print("=" * 80)

    try:
        # Load images
        print("Loading images...")
        image1 = load_and_preprocess_image(args.image1)
        image2 = load_and_preprocess_image(args.image2)
        print(f"Image1 size: {image1.shape}")
        print(f"Image2 size: {image2.shape}")

        # Import modules
        from harris import get_harris_corners
        from descriptors import get_descriptors
        from matching import match_features
        from ransac import ransac

        # Stage 1: Detect Harris corners
        print("\n" + "-" * 60)
        print("Detecting Harris corners...")
        corners1, response1 = get_harris_corners(image1, args.sigma, args.k, args.threshold)
        corners2, response2 = get_harris_corners(image2, args.sigma, args.k, args.threshold)
        print(f"Image1 detected {len(corners1)} corners")
        print(f"Image2 detected {len(corners2)} corners")

        # Stage 2: Extract descriptors
        print("\n" + "-" * 60)
        print("Extracting feature descriptors...")
        descriptors1 = get_descriptors(image1, corners1, args.sigma)
        descriptors2 = get_descriptors(image2, corners2, args.sigma)
        print(f"Descriptor dimension: {descriptors1.shape[1]}")

        # Stage 3: Match features
        print("\n" + "-" * 60)
        print("Matching features...")
        matches = match_features(descriptors1, descriptors2, args.ratio_threshold)
        print(f"Raw matches: {len(matches)}")

        # Stage 4: RANSAC outlier rejection
        print("\n" + "-" * 60)
        print("RANSAC outlier rejection...")
        transform, inlier_matches, inlier_mask = ransac(
            corners1, corners2, matches,
            args.ransac_iterations,
            args.ransac_threshold,
            args.transform
        )

        print(f"Inlier matches: {len(inlier_matches)}")
        print(f"Inlier ratio: {len(inlier_matches)/len(matches):.2%}")

        # Visualize results
        print("\n" + "-" * 60)
        print("Visualizing results...")
        visualize_results(image1, image2, corners1, corners2, matches, inlier_matches)

        print("\n" + "*" * 50)
        print("Program execution completed")
        print("*" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
