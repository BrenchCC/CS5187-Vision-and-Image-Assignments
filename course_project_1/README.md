# 课程项目1: 图像特征匹配与异常值剔除

## 项目概述
本项目实现了图像特征匹配的完整流程，包括Harris角点检测、特征描述、特征匹配和RANSAC异常值剔除。

## 功能特性
- **Harris角点检测**: 计算图像梯度、结构张量矩阵、高斯窗口和局部最大值检测
- **特征描述符**: 简化版SIFT描述符，具有旋转和尺度不变性
- **特征匹配**: 平方差和(SSD)匹配，配合Lowe比率测试
- **异常值剔除**: RANSAC算法，支持单应性和仿射变换

## 文件结构
```
course_project_1/
├── src/
│   ├── __init__.py
│   ├── harris.py          # Harris角点检测
│   ├── descriptors.py    # 特征描述符
│   ├── matching.py       # 特征匹配
│   ├── ransac.py         # RANSAC算法
│   └── main.py           # 主程序
├── data/                 # 测试图像数据 (需要创建)
└── requirements.txt      # 依赖包
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 运行主程序
```bash
cd src
python main.py <图像1路径> <图像2路径> [可选参数]
```

### 可选参数
- `--sigma`: 高斯平滑参数 (默认: 1.0)
- `--k`: Harris响应函数常数 (默认: 0.04)
- `--threshold`: 角点检测阈值 (默认: 0.01)
- `--ratio-threshold`: Lowe比率阈值 (默认: 0.75)
- `--ransac-iterations`: RANSAC迭代次数 (默认: 1000)
- `--ransac-threshold`: RANSAC内点阈值 (默认: 5.0)
- `--transform`: 变换类型 (homography 或 affine, 默认: homography)

### 示例
```bash
python main.py ../data/image1.jpg ../data/image2.jpg
python main.py ../data/img1.png ../data/img2.png --transform affine --sigma 1.5
```

## 模块说明

### harris.py
```python
from harris import get_harris_corners
corners, response = get_harris_corners(image, sigma=1.0, k=0.04, threshold=0.01)
```

### descriptors.py
```python
from descriptors import get_descriptors
descriptors = get_descriptors(image, corners, sigma=1.0)
```

### matching.py
```python
from matching import match_features
matches = match_features(descriptors1, descriptors2, ratio_threshold=0.75)
```

### ransac.py
```python
from ransac import ransac
transform, inlier_matches, inlier_mask = ransac(
    corners1, corners2, matches,
    num_iterations=1000,
    inlier_threshold=5.0,
    transform_type='homography'
)
```

## 输出结果
- 角点检测数量
- 描述符维度
- 原始匹配数量
- 内点匹配数量
- 内点比例
- 可视化匹配结果 (红色: 原始匹配, 绿色: 内点匹配)
