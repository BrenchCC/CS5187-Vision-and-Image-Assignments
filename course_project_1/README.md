# CS5187 Course Project 1

## Image Feature Matching and Outlier Rejection

This project implements a complete classical feature matching pipeline from scratch for robust image correspondence under real-world challenges. The system includes Harris corner detection, local descriptor extraction, descriptor matching with Lowe's ratio test, and RANSAC-based outlier rejection. It was built for the CS5187 vision assignment on feature detection and matching in challenging scenes.

## Project Goals
- implement Harris corner detection from scratch
- design a simplified invariant local descriptor
- match descriptors using SSD and Lowe's ratio test
- reject outliers with RANSAC and estimate geometric transformation
- evaluate the pipeline on multiple challenging image-pair scenarios

## Implemented Pipeline
The pipeline contains the following stages:

1. Read and preprocess input images as grayscale images
2. Compute image gradients with Sobel filters
3. Detect Harris corners using the structure tensor and local maxima
4. Extract simplified SIFT-style descriptors from local patches
5. Match descriptors with SSD and Lowe's ratio filtering
6. Estimate a homography with RANSAC
7. Save a side-by-side visualization with inlier match lines

## Repository Structure
```text
course_project_1/
├── data/                       # Downloaded and converted images
├── docs/                       # Assignment description and notes
├── results/                    # Generated matching result images
├── src/
│   ├── main.py                 # Main pipeline entry point
│   ├── harris.py               # Harris corner detector
│   ├── descriptors.py          # Local descriptor extraction
│   ├── matching.py             # SSD matching + ratio test
│   ├── ransac.py               # RANSAC geometric verification
│   ├── convert_dataset_images.py
│   └── utils/
│       ├── convolution.py
│       └── image_utils.py
├── submission/                 # Organized deliverables
├── prepare_submission.sh       # Prepare submission assets
├── run_batch_pipeline.sh       # Batch reproduction script
├── requirements.txt
└── README.md
```

## Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation
Some downloaded benchmark images are stored as `ppm` or `pgm`. Convert them to `png` before running experiments:

```bash
cd src
python convert_dataset_images.py --input-dir ../data --output-dir ../data/converted
```

The project also copies converted images into the original scenario folders for easier use.

## Run a Single Image Pair
Run the pipeline on one image pair:

```bash
cd src
python main.py <image1_path> <image2_path> [optional_arguments]
```

Example:

```bash
cd src
python main.py ../data/blur_bikes/img1.png ../data/blur_bikes/img4.png --output ../results/blur_bikes_matches.png --no-display
```

## Main Arguments
- `--sigma`: Gaussian smoothing parameter, default `1.0`
- `--k`: Harris response constant, default `0.04`
- `--threshold`: corner detection threshold, default `0.01`
- `--ratio-threshold`: Lowe ratio threshold, default `0.75`
- `--ransac-iterations`: number of RANSAC iterations, default `1000`
- `--ransac-threshold`: inlier threshold in pixels, default `5.0`
- `--transform`: `homography` or `affine`
- `--output`: output path for the visualization image
- `--no-display`: disable interactive plotting

## Batch Reproduction
Run all prepared scenarios at once:

```bash
./run_batch_pipeline.sh
```

This script runs the current selected scenarios and writes result images into `results/`.

## Selected Evaluation Scenarios
The current project setup contains five evaluated scenarios:
- blur / defocus
- illumination change
- perspective / viewpoint change
- scale / zoom change
- severe clutter

## Current Outputs
Generated result images are stored in `results/`, for example:
- `blur_bikes_matches.png`
- `illumination_leuven_matches.png`
- `perspective_graf_matches.png`
- `scale_boat_matches.png`
- `clutter_box_matches.png`

## Submission Package
The organized deliverables are stored in `submission/`, including:
- `code/`
- `input_images/`
- `input_images.zip`
- `output_images/`
- `report/project_report.tex`
- `report/project_report.pdf`

## Notes
The assignment text contains a requirement mismatch: one section asks to choose 5 scenarios out of 10, while the deliverables section mentions all 10 scenarios. The current project organization follows the explicit “choose 5 scenarios” instruction and prepares a consistent 5-scenario submission package.
