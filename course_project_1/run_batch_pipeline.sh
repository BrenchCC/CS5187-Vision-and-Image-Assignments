#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${ROOT_DIR}/src"
DATA_DIR="${ROOT_DIR}/data"
RESULTS_DIR="${ROOT_DIR}/results"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cs_hw}"

mkdir -p "${RESULTS_DIR}"

echo "======================================================================"
echo "Running feature matching batch pipeline"
echo "======================================================================"
echo "Using Conda environment: ${CONDA_ENV_NAME}"

run_case() {
    local case_name="$1"
    local image1="$2"
    local image2="$3"
    local transform="$4"

    echo "----------------------------------------------------------------------"
    echo "Processing case: ${case_name}"
    echo "----------------------------------------------------------------------"

    (
        cd "${SRC_DIR}"
        conda run -n "${CONDA_ENV_NAME}" python main.py \
            "${image1}" \
            "${image2}" \
            --transform "${transform}" \
            --output "${RESULTS_DIR}/${case_name}_matches.png" \
            --no-display
    )
}

run_case \
    "blur_bikes" \
    "${DATA_DIR}/blur_bikes/img1.png" \
    "${DATA_DIR}/blur_bikes/img4.png" \
    "homography"

run_case \
    "illumination_leuven" \
    "${DATA_DIR}/illumination_leuven/img1.png" \
    "${DATA_DIR}/illumination_leuven/img4.png" \
    "homography"

run_case \
    "perspective_graf" \
    "${DATA_DIR}/perspective_graf/img1.png" \
    "${DATA_DIR}/perspective_graf/img3.png" \
    "homography"

run_case \
    "scale_boat" \
    "${DATA_DIR}/scale_boat/img1.png" \
    "${DATA_DIR}/scale_boat/img3.png" \
    "homography"

run_case \
    "clutter_box" \
    "${DATA_DIR}/clutter_box/box.png" \
    "${DATA_DIR}/clutter_box/box_in_scene.png" \
    "homography"

echo "**********************************************************************"
echo "Batch pipeline completed"
echo "Results saved in: ${RESULTS_DIR}"
echo "**********************************************************************"
