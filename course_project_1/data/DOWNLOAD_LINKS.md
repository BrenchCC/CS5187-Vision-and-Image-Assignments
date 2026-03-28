# Course Project 1 Image Sources

This directory contains 5 image pairs collected from public internet sources for feature matching evaluation.

## Selected Scenarios

### 1. Scale / Zoom
- Scenario: `scale_boat`
- Dataset/source page: https://www.robots.ox.ac.uk/~vgg/research/affine/
- Direct download archive: https://thor.robots.ox.ac.uk/affine/boat.tar.gz
- Selected files from archive:
  - `img1.pgm`
  - `img3.pgm`
- Local files:
  - `course_project_1/data/scale_boat/img1.pgm`
  - `course_project_1/data/scale_boat/img3.pgm`
- Note: This pair includes noticeable zoom/scale change and some rotation.

### 2. Perspective / Viewpoint Change
- Scenario: `perspective_graf`
- Dataset/source page: https://www.robots.ox.ac.uk/~vgg/research/affine/
- Direct download archive: https://thor.robots.ox.ac.uk/affine/graf.tar.gz
- Selected files from archive:
  - `img1.ppm`
  - `img3.ppm`
- Local files:
  - `course_project_1/data/perspective_graf/img1.ppm`
  - `course_project_1/data/perspective_graf/img3.ppm`

### 3. Illumination Change
- Scenario: `illumination_leuven`
- Dataset/source page: https://www.robots.ox.ac.uk/~vgg/research/affine/
- Direct download archive: https://thor.robots.ox.ac.uk/affine/leuven.tar.gz
- Selected files from archive:
  - `img1.ppm`
  - `img4.ppm`
- Local files:
  - `course_project_1/data/illumination_leuven/img1.ppm`
  - `course_project_1/data/illumination_leuven/img4.ppm`

### 4. Defocus / Blur
- Scenario: `blur_bikes`
- Dataset/source page: https://www.robots.ox.ac.uk/~vgg/research/affine/
- Direct download archive: https://thor.robots.ox.ac.uk/affine/bikes.tar.gz
- Selected files from archive:
  - `img1.ppm`
  - `img4.ppm`
- Local files:
  - `course_project_1/data/blur_bikes/img1.ppm`
  - `course_project_1/data/blur_bikes/img4.ppm`

### 5. Severe Clutter
- Scenario: `clutter_box`
- Source repository: https://github.com/opencv/opencv/tree/4.x/samples/data
- Direct download links:
  - https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/box.png
  - https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/box_in_scene.png
- Local files:
  - `course_project_1/data/clutter_box/box.png`
  - `course_project_1/data/clutter_box/box_in_scene.png`
- Note: This pair is widely used for local feature matching with cluttered background.

## Re-download Commands

```bash
cd course_project_1/data

curl -L -o _archives/boat.tar.gz https://thor.robots.ox.ac.uk/affine/boat.tar.gz
tar -xzf _archives/boat.tar.gz -C scale_boat img1.pgm img3.pgm

curl -L -o _archives/graf.tar.gz https://thor.robots.ox.ac.uk/affine/graf.tar.gz
tar -xzf _archives/graf.tar.gz -C perspective_graf img1.ppm img3.ppm

curl -L -o _archives/leuven.tar.gz https://thor.robots.ox.ac.uk/affine/leuven.tar.gz
tar -xzf _archives/leuven.tar.gz -C illumination_leuven img1.ppm img4.ppm

curl -L -o _archives/bikes.tar.gz https://thor.robots.ox.ac.uk/affine/bikes.tar.gz
tar -xzf _archives/bikes.tar.gz -C blur_bikes img1.ppm img4.ppm

curl -L -o clutter_box/box.png https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/box.png
curl -L -o clutter_box/box_in_scene.png https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/box_in_scene.png
```
