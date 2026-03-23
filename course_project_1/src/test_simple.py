#!/usr/bin/env python3
"""
Simple Test Script - Verify Basic Functionality

This script verifies the functionality of each module using synthetic test images.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utilities from utils
from utils.image_utils import (
    create_test_image_with_corners,
    create_shifted_test_image
)


def test_harris():
    """Test Harris corner detector"""
    print("Testing Harris corner detector...")
    from harris import get_harris_corners

    image = create_test_image_with_corners()
    corners, response = get_harris_corners(image, sigma=1.0, k=0.04, threshold=0.01)

    print(f"  Detected {len(corners)} corners")
    print(f"  Harris response shape: {response.shape}")
    print(f"  Harris response range: [{np.min(response):.3f}, {np.max(response):.3f}]")

    assert len(corners) > 0, "Should detect at least one corner"
    print("  Harris corner detector: PASSED\n")
    return True


def test_descriptors():
    """Test feature descriptors"""
    print("Testing feature descriptors...")
    from harris import get_harris_corners
    from descriptors import get_descriptors

    image = create_test_image_with_corners()
    corners, _ = get_harris_corners(image, sigma=1.0, k=0.04, threshold=0.01)

    # If too many corners, only take first 10 for testing
    if len(corners) > 10:
        corners = corners[:10]

    descriptors = get_descriptors(image, corners, sigma=1.0)

    print(f"  Descriptors shape: {descriptors.shape}")
    print(f"  Number of descriptors: {descriptors.shape[0]}")
    print(f"  Dimension per descriptor: {descriptors.shape[1]}")

    assert len(corners) == descriptors.shape[0], "Number of descriptors should equal number of corners"
    assert descriptors.shape[1] > 0, "Descriptor dimension should be greater than 0"
    print("  Feature descriptors: PASSED\n")
    return True


def test_matching():
    """Test feature matching"""
    print("Testing feature matching...")
    from harris import get_harris_corners
    from descriptors import get_descriptors
    from matching import match_features

    image1 = create_test_image_with_corners()
    image2 = create_shifted_test_image(image1, dx=3, dy=3)

    corners1, _ = get_harris_corners(image1, sigma=1.0, k=0.04, threshold=0.02)
    corners2, _ = get_harris_corners(image2, sigma=1.0, k=0.04, threshold=0.02)

    # Limit number
    if len(corners1) > 20:
        corners1 = corners1[:20]
    if len(corners2) > 20:
        corners2 = corners2[:20]

    descriptors1 = get_descriptors(image1, corners1, sigma=1.0)
    descriptors2 = get_descriptors(image2, corners2, sigma=1.0)

    matches = match_features(descriptors1, descriptors2, ratio_threshold=0.85)

    print(f"  Image1 corner count: {len(corners1)}")
    print(f"  Image2 corner count: {len(corners2)}")
    print(f"  Match count: {len(matches)}")

    print("  Feature matching: PASSED\n")
    return True


def test_ransac():
    """Test RANSAC algorithm"""
    print("Testing RANSAC algorithm...")
    from harris import get_harris_corners
    from descriptors import get_descriptors
    from matching import match_features
    from ransac import ransac

    image1 = create_test_image_with_corners()
    image2 = create_shifted_test_image(image1, dx=5, dy=5)

    corners1, _ = get_harris_corners(image1, sigma=1.0, k=0.04, threshold=0.02)
    corners2, _ = get_harris_corners(image2, sigma=1.0, k=0.04, threshold=0.02)

    # Limit number
    if len(corners1) > 20:
        corners1 = corners1[:20]
    if len(corners2) > 20:
        corners2 = corners2[:20]

    descriptors1 = get_descriptors(image1, corners1, sigma=1.0)
    descriptors2 = get_descriptors(image2, corners2, sigma=1.0)

    matches = match_features(descriptors1, descriptors2, ratio_threshold=0.85)

    if len(matches) >= 4:
        transform, inlier_matches, inlier_mask = ransac(
            corners1, corners2, matches,
            num_iterations=200,
            inlier_threshold=10.0,
            transform_type='homography'
        )

        print(f"  Raw matches: {len(matches)}")
        print(f"  Inlier matches: {len(inlier_matches)}")
        print(f"  Inlier ratio: {len(inlier_matches)/len(matches):.2%}")

        if transform is not None:
            print(f"  Transform matrix:")
            print(f"    [[{transform[0,0]:.4f}, {transform[0,1]:.4f}, {transform[0,2]:.4f}],")
            print(f"     [{transform[1,0]:.4f}, {transform[1,1]:.4f}, {transform[1,2]:.4f}],")
            print(f"     [{transform[2,0]:.4f}, {transform[2,1]:.4f}, {transform[2,2]:.4f}]]")

        print("  RANSAC algorithm: PASSED\n")
        return True
    else:
        print(f"  Insufficient matches (need >=4, got {len(matches)}), skipping RANSAC test")
        print("  RANSAC algorithm: SKIPPED\n")
        return True


def main():
    """Main test function"""
    print("=" * 80)
    print("Course Project 1 - Functionality Test")
    print("=" * 80)

    tests = [
        ("Harris corner detector", test_harris),
        ("Feature descriptors", test_descriptors),
        ("Feature matching", test_matching),
        ("RANSAC algorithm", test_ransac)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"  Error: {e}")
            import traceback
            print(f"  {traceback.format_exc()}\n")

    print("=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = 0
    for test_name, result, error in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")
    print("=" * 80)

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
