#!/usr/bin/env python3
"""
Convert dataset images into standard image formats.

This script recursively scans an input directory, finds PPM/PGM images,
and converts them into PNG files while preserving the folder structure.
"""

import argparse
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm


logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {
    ".ppm",
    ".pgm"
}


def parse_args():
    """
    Parse command line arguments.

    Parameters:
        None

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description = "Convert PPM/PGM dataset images into PNG files."
    )

    parser.add_argument(
        "--input-dir",
        type = str,
        default = "../data",
        help = "Directory containing downloaded dataset images."
    )

    parser.add_argument(
        "--output-dir",
        type = str,
        default = "../data/converted",
        help = "Directory for converted PNG files."
    )

    parser.add_argument(
        "--overwrite",
        action = "store_true",
        help = "Overwrite existing output files."
    )

    return parser.parse_args()


def collect_source_images(input_dir):
    """
    Collect supported source images recursively.

    Parameters:
        input_dir (Path): Directory to scan.

    Returns:
        image_paths (list[Path]): Supported image file paths.
    """
    image_paths = []

    for image_path in sorted(input_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(image_path)

    return image_paths


def convert_image(
    image_path,
    input_dir,
    output_dir,
    overwrite = False
):
    """
    Convert one image into PNG format.

    Parameters:
        image_path (Path): Source image path.
        input_dir (Path): Root input directory.
        output_dir (Path): Root output directory.
        overwrite (bool): Whether to overwrite existing files.

    Returns:
        output_path (Path): Generated PNG path.
    """
    relative_path = image_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix(".png")
    output_path.parent.mkdir(parents = True, exist_ok = True)

    if output_path.exists() and not overwrite:
        logger.info("Skip existing file: %s", output_path)
        return output_path

    with Image.open(image_path) as image:
        image.save(output_path, format = "PNG")

    logger.info("Converted %s -> %s", image_path, output_path)
    return output_path


def main():
    """
    Run the dataset image conversion workflow.

    Parameters:
        None

    Returns:
        None
    """
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    logger.info("=" * 80)
    logger.info("Dataset image conversion started")
    logger.info("=" * 80)
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)

    image_paths = collect_source_images(input_dir)

    logger.info("-" * 60)
    logger.info("Found %d PPM/PGM images for conversion", len(image_paths))
    logger.info("-" * 60)

    for image_path in tqdm(image_paths, desc = "Converting images"):
        convert_image(
            image_path = image_path,
            input_dir = input_dir,
            output_dir = output_dir,
            overwrite = args.overwrite
        )

    logger.info("*" * 50)
    logger.info("Dataset image conversion completed")
    logger.info("*" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
