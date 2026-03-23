"""
课程项目1: 图像特征匹配与异常值剔除

该包包含以下模块:
- harris: Harris角点检测器
- descriptors: 特征描述符提取
- matching: 特征匹配算法
- ransac: RANSAC异常值剔除
"""

__version__ = "1.0.0"
__author__ = "Brench"

# 导出主要函数
from .harris import get_harris_corners
from .descriptors import get_descriptors
from .matching import match_features
from .ransac import ransac
