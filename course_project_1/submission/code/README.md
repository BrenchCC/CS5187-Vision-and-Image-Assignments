# Course Project 1: Image Feature Matching and Outlier Rejection

## Overview
This project implements a complete local feature matching pipeline from scratch using Python and NumPy-style components. The pipeline includes Harris corner detection, local descriptor extraction, feature matching with Lowe's ratio test, and RANSAC-based outlier rejection.

## Main Features
- **Harris corner detection**: image gradients, structure tensor construction, Gaussian smoothing, and local maximum detection
- **Local descriptors**: simplified SIFT-style descriptors based on gradient orientation histograms
- **Feature matching**: SSD nearest-neighbor matching with Lowe's ratio test
- **Outlier rejection**: RANSAC for robust geometric verification with homography estimation
- **Visualization output**: side-by-side result images with inlier match lines after RANSAC

## File Structure
```text
code/
├── src/
│   ├── main.py
│   ├── harris.py
│   ├── descriptors.py
│   ├── matching.py
│   ├── ransac.py
│   ├── convert_dataset_images.py
│   └── utils/
│       ├── convolution.py
│       └── image_utils.py
├── run_batch_pipeline.sh
├── prepare_submission.sh
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Prepare Dataset Images
If the dataset contains `ppm` or `pgm` files, convert them to `png` first:

```bash
cd src
python convert_dataset_images.py --input-dir ../data --output-dir ../data/converted
```

This script scans the input directory recursively, converts `ppm/pgm` images to `png`, and preserves the original folder structure.

## Run the Main Pipeline
```bash
cd src
python main.py <image1_path> <image2_path> [optional_arguments]
```

## Common Arguments
- `--sigma`: Gaussian smoothing parameter, default `1.0`
- `--k`: Harris response constant, default `0.04`
- `--threshold`: corner detection threshold, default `0.01`
- `--ratio-threshold`: Lowe ratio threshold, default `0.75`
- `--ransac-iterations`: number of RANSAC iterations, default `1000`
- `--ransac-threshold`: RANSAC inlier threshold in pixels, default `5.0`
- `--transform`: transformation model, either `homography` or `affine`
- `--output`: save the visualization image to a file
- `--no-display`: disable interactive display and only save the output image

## Examples
```bash
cd src
python convert_dataset_images.py --input-dir ../data --output-dir ../data/converted
python main.py ../data/blur_bikes/img1.png ../data/blur_bikes/img4.png --output ../results/blur_bikes_matches.png --no-display
python main.py ../data/perspective_graf/img1.png ../data/perspective_graf/img3.png --transform homography --output ../results/perspective_graf_matches.png --no-display
```

## Batch Reproduction
To reproduce all prepared scenario results in one command:

```bash
cd ..
./run_batch_pipeline.sh
```

This script runs the pipeline on the selected scenarios and saves the output images into `results/`.

## Core Modules

### `harris.py`
Implements Harris corner detection:

```python
from harris import get_harris_corners

corners, response = get_harris_corners(
    image,
    sigma = 1.0,
    k = 0.04,
    threshold = 0.01
)
```

### `descriptors.py`
Extracts simplified SIFT-style descriptors:

```python
from descriptors import get_descriptors

descriptors = get_descriptors(image, corners, sigma = 1.0)
```

### `matching.py`
Matches descriptors with SSD and Lowe's ratio test:

```python
from matching import match_features

matches = match_features(descriptors1, descriptors2, ratio_threshold = 0.75)
```

### `ransac.py`
Rejects outliers using RANSAC:

```python
from ransac import ransac

transform, inlier_matches, inlier_mask = ransac(
    corners1,
    corners2,
    matches,
    num_iterations = 1000,
    inlier_threshold = 5.0,
    transform_type = "homography"
)
```

## Output
The program reports:
- number of detected corners in each image
- descriptor dimensionality
- number of raw matches
- number of inlier matches after RANSAC
- inlier ratio
- saved visualization image showing matched inlier correspondences
