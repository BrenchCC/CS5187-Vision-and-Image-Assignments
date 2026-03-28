#!/usr/bin/env python3
"""
Course Project 1: Image Feature Matching and Outlier Rejection.

This program runs the complete image feature matching pipeline:
1. Read and preprocess input images
2. Detect Harris corners
3. Extract local descriptors
4. Match descriptors
5. Reject outliers with RANSAC
6. Save or display the match visualization
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(os.getcwd())

from descriptors import get_descriptors
from harris import get_harris_corners
from matching import match_features
from ransac import ransac
from utils.image_utils import load_and_preprocess_image


logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Parameters:
        None

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Image Feature Matching and Outlier Rejection"
    )

    parser.add_argument(
        "image1",
        help = "Path to the first image"
    )
    parser.add_argument(
        "image2",
        help = "Path to the second image"
    )
    parser.add_argument(
        "--sigma",
        type = float,
        default = 1.0,
        help = "Gaussian smoothing parameter"
    )
    parser.add_argument(
        "--k",
        type = float,
        default = 0.04,
        help = "Harris response constant"
    )
    parser.add_argument(
        "--threshold",
        type = float,
        default = 0.01,
        help = "Corner detection threshold"
    )
    parser.add_argument(
        "--ratio-threshold",
        type = float,
        default = 0.75,
        help = "Lowe ratio threshold"
    )
    parser.add_argument(
        "--ransac-iterations",
        type = int,
        default = 1000,
        help = "Number of RANSAC iterations"
    )
    parser.add_argument(
        "--ransac-threshold",
        type = float,
        default = 5.0,
        help = "RANSAC inlier threshold in pixels"
    )
    parser.add_argument(
        "--transform",
        choices = ["homography", "affine"],
        default = "homography",
        help = "Transformation model used by RANSAC"
    )
    parser.add_argument(
        "--output",
        type = str,
        default = "",
        help = "Optional output image path for the match visualization"
    )
    parser.add_argument(
        "--no-display",
        action = "store_true",
        help = "Disable interactive display and only save output"
    )

    return parser.parse_args()


def create_combined_image(image1, image2):
    """
    Create a side-by-side visualization canvas.

    Parameters:
        image1 (np.ndarray): First grayscale image.
        image2 (np.ndarray): Second grayscale image.

    Returns:
        combined (np.ndarray): Combined visualization image.
        image1_width (int): Width of the first image.
    """
    image1_height, image1_width = image1.shape
    image2_height, image2_width = image2.shape

    combined = np.zeros(
        (
            max(image1_height, image2_height),
            image1_width + image2_width
        ),
        dtype = np.uint8
    )
    combined[:image1_height, :image1_width] = image1
    combined[:image2_height, image1_width:image1_width + image2_width] = image2

    return combined, image1_width


def save_matching_visualization(
    image1,
    image2,
    corners1,
    corners2,
    matches,
    inlier_matches,
    output_path = "",
    show_plot = True
):
    """
    Save or display the feature matching visualization.

    Parameters:
        image1 (np.ndarray): First grayscale image.
        image2 (np.ndarray): Second grayscale image.
        corners1 (np.ndarray): Corner coordinates from image 1.
        corners2 (np.ndarray): Corner coordinates from image 2.
        matches (np.ndarray): All matched feature pairs.
        inlier_matches (np.ndarray): Inlier matched feature pairs.
        output_path (str): Output image path.
        show_plot (bool): Whether to display the figure interactively.

    Returns:
        None
    """
    combined, image1_width = create_combined_image(image1, image2)

    figure = plt.figure(figsize = (15, 8))
    plt.imshow(combined, cmap = "gray")

    for index1, index2 in matches:
        x1, y1 = corners1[index1][1], corners1[index1][0]
        x2, y2 = corners2[index2][1] + image1_width, corners2[index2][0]
        plt.plot([x1, x2], [y1, y2], "r-", linewidth = 0.5, alpha = 0.25)

    for index1, index2 in inlier_matches:
        x1, y1 = corners1[index1][1], corners1[index1][0]
        x2, y2 = corners2[index2][1] + image1_width, corners2[index2][0]
        plt.plot([x1, x2], [y1, y2], "g-", linewidth = 1.0, alpha = 0.85)

    if len(corners1) > 0:
        plt.scatter(
            corners1[:, 1],
            corners1[:, 0],
            c = "b",
            s = 15,
            alpha = 0.45,
            label = "Image 1 corners"
        )

    if len(corners2) > 0:
        plt.scatter(
            corners2[:, 1] + image1_width,
            corners2[:, 0],
            c = "orange",
            s = 15,
            alpha = 0.45,
            label = "Image 2 corners"
        )

    plt.title(
        f"Feature Matching Results: {len(matches)} total matches, "
        f"{len(inlier_matches)} inliers"
    )
    plt.legend(loc = "upper right")
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents = True, exist_ok = True)
        figure.savefig(output_file, dpi = 200, bbox_inches = "tight")
        logger.info("Saved visualization to %s", output_file)

    if show_plot:
        plt.show()

    plt.close(figure)


def run_pipeline(args):
    """
    Run the full image matching pipeline.

    Parameters:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        summary (dict): Pipeline result summary.
    """
    logger.info("=" * 80)
    logger.info("Image feature matching pipeline started")
    logger.info("=" * 80)
    logger.info("Image 1: %s", args.image1)
    logger.info("Image 2: %s", args.image2)

    image1 = load_and_preprocess_image(args.image1)
    image2 = load_and_preprocess_image(args.image2)

    logger.info("Image 1 shape: %s", image1.shape)
    logger.info("Image 2 shape: %s", image2.shape)

    logger.info("-" * 60)
    logger.info("Detecting Harris corners")
    logger.info("-" * 60)
    corners1, _ = get_harris_corners(image1, args.sigma, args.k, args.threshold)
    corners2, _ = get_harris_corners(image2, args.sigma, args.k, args.threshold)
    logger.info("Detected %d corners in image 1", len(corners1))
    logger.info("Detected %d corners in image 2", len(corners2))

    logger.info("-" * 60)
    logger.info("Extracting descriptors")
    logger.info("-" * 60)
    descriptors1 = get_descriptors(image1, corners1, args.sigma)
    descriptors2 = get_descriptors(image2, corners2, args.sigma)

    if len(descriptors1) == 0 or len(descriptors2) == 0:
        raise RuntimeError("Descriptor extraction returned an empty result.")

    logger.info("Descriptor dimension: %d", descriptors1.shape[1])

    logger.info("-" * 60)
    logger.info("Matching descriptors")
    logger.info("-" * 60)
    matches = match_features(descriptors1, descriptors2, args.ratio_threshold)
    if matches.size == 0:
        matches = np.empty((0, 2), dtype = int)
    logger.info("Raw matches: %d", len(matches))

    required_matches = 4 if args.transform == "homography" else 3
    inlier_matches = np.empty((0, 2), dtype = int)
    transform = None

    logger.info("-" * 60)
    logger.info("Running RANSAC")
    logger.info("-" * 60)
    if len(matches) >= required_matches:
        transform, inlier_matches, _ = ransac(
            corners1,
            corners2,
            matches,
            args.ransac_iterations,
            args.ransac_threshold,
            args.transform
        )
        if inlier_matches.size == 0:
            inlier_matches = np.empty((0, 2), dtype = int)
    else:
        logger.warning(
            "Skipped RANSAC because %d matches are fewer than the %d required "
            "for %s.",
            len(matches),
            required_matches,
            args.transform
        )

    inlier_ratio = 0.0
    if len(matches) > 0:
        inlier_ratio = len(inlier_matches) / len(matches)

    logger.info("Inlier matches: %d", len(inlier_matches))
    logger.info("Inlier ratio: %.2f%%", inlier_ratio * 100.0)

    logger.info("-" * 60)
    logger.info("Rendering visualization")
    logger.info("-" * 60)
    save_matching_visualization(
        image1 = image1,
        image2 = image2,
        corners1 = corners1,
        corners2 = corners2,
        matches = matches,
        inlier_matches = inlier_matches,
        output_path = args.output,
        show_plot = not args.no_display
    )

    logger.info("*" * 50)
    logger.info("Pipeline completed")
    logger.info("*" * 50)

    return {
        "transform": transform,
        "corners1": len(corners1),
        "corners2": len(corners2),
        "matches": len(matches),
        "inliers": len(inlier_matches),
        "inlier_ratio": inlier_ratio
    }


def main():
    """
    Entry point of the program.

    Parameters:
        None

    Returns:
        int: Process exit code.
    """
    args = parse_args()

    try:
        run_pipeline(args)
        return 0
    except Exception as error:
        logger.exception("Pipeline failed: %s", error)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    raise SystemExit(main())
