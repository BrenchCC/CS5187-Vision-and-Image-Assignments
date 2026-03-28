#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_DIR="${ROOT_DIR}/submission"
INPUT_DIR="${SUBMISSION_DIR}/input_images"
OUTPUT_DIR="${SUBMISSION_DIR}/output_images"
REPORT_DIR="${SUBMISSION_DIR}/report"
FIGURE_DIR="${REPORT_DIR}/figures"
ZIP_PATH="${SUBMISSION_DIR}/input_images.zip"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}" "${FIGURE_DIR}"

copy_file() {
    local source_path="$1"
    local target_path="$2"

    cp "${source_path}" "${target_path}"
    echo "Copied ${source_path} -> ${target_path}"
}

copy_file \
    "${ROOT_DIR}/data/blur_bikes/img1.png" \
    "${INPUT_DIR}/scenario_01_blur_inputA.png"
copy_file \
    "${ROOT_DIR}/data/blur_bikes/img4.png" \
    "${INPUT_DIR}/scenario_01_blur_inputB.png"
copy_file \
    "${ROOT_DIR}/results/blur_bikes_matches.png" \
    "${OUTPUT_DIR}/scenario_01_blur_output.png"

copy_file \
    "${ROOT_DIR}/data/illumination_leuven/img1.png" \
    "${INPUT_DIR}/scenario_02_illumination_inputA.png"
copy_file \
    "${ROOT_DIR}/data/illumination_leuven/img4.png" \
    "${INPUT_DIR}/scenario_02_illumination_inputB.png"
copy_file \
    "${ROOT_DIR}/results/illumination_leuven_matches.png" \
    "${OUTPUT_DIR}/scenario_02_illumination_output.png"

copy_file \
    "${ROOT_DIR}/data/perspective_graf/img1.png" \
    "${INPUT_DIR}/scenario_03_perspective_inputA.png"
copy_file \
    "${ROOT_DIR}/data/perspective_graf/img3.png" \
    "${INPUT_DIR}/scenario_03_perspective_inputB.png"
copy_file \
    "${ROOT_DIR}/results/perspective_graf_matches.png" \
    "${OUTPUT_DIR}/scenario_03_perspective_output.png"

copy_file \
    "${ROOT_DIR}/data/scale_boat/img1.png" \
    "${INPUT_DIR}/scenario_04_scale_inputA.png"
copy_file \
    "${ROOT_DIR}/data/scale_boat/img3.png" \
    "${INPUT_DIR}/scenario_04_scale_inputB.png"
copy_file \
    "${ROOT_DIR}/results/scale_boat_matches.png" \
    "${OUTPUT_DIR}/scenario_04_scale_output.png"

copy_file \
    "${ROOT_DIR}/data/clutter_box/box.png" \
    "${INPUT_DIR}/scenario_05_clutter_inputA.png"
copy_file \
    "${ROOT_DIR}/data/clutter_box/box_in_scene.png" \
    "${INPUT_DIR}/scenario_05_clutter_inputB.png"
copy_file \
    "${ROOT_DIR}/results/clutter_box_matches.png" \
    "${OUTPUT_DIR}/scenario_05_clutter_output.png"

copy_file \
    "${ROOT_DIR}/results/blur_bikes_matches.png" \
    "${FIGURE_DIR}/scenario_01_blur_output.png"
copy_file \
    "${ROOT_DIR}/results/illumination_leuven_matches.png" \
    "${FIGURE_DIR}/scenario_02_illumination_output.png"
copy_file \
    "${ROOT_DIR}/results/perspective_graf_matches.png" \
    "${FIGURE_DIR}/scenario_03_perspective_output.png"
copy_file \
    "${ROOT_DIR}/results/scale_boat_matches.png" \
    "${FIGURE_DIR}/scenario_04_scale_output.png"
copy_file \
    "${ROOT_DIR}/results/clutter_box_matches.png" \
    "${FIGURE_DIR}/scenario_05_clutter_output.png"

rm -f "${ZIP_PATH}"
(
    cd "${SUBMISSION_DIR}"
    zip -rq "$(basename "${ZIP_PATH}")" "$(basename "${INPUT_DIR}")"
)

echo "Submission assets prepared in ${SUBMISSION_DIR}"
