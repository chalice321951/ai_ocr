"""
基于 HSV 颜色空间的 ROI 检测器
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColorDetectorConfig:
    """颜色检测配置"""
    hsv_lower: Tuple[int, int, int] = (0, 100, 80)
    hsv_upper: Tuple[int, int, int] = (15, 255, 255)
    min_area: int = 1000
    padding: int = 10
    max_height_ratio: float = 0.15  # 高度占画面最大比例（只取窄条）
    min_width_ratio: float = 0.3  # 宽度占画面最小比例（排除零散红点）


class ColorDetector:
    """基于 HSV 颜色空间的 ROI 检测器"""

    def __init__(self, config: ColorDetectorConfig):
        self.config = config

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测红色导航条。
        原理：横向膨胀把碎片连成一整块，再用宽高比过滤找到又宽又窄的条。

        参数:
            frame: BGR 图像 (H, W, C)

        返回:
            (x, y, w, h) 或 None
        """
        h_frame, w_frame = frame.shape[:2]

        # 1. HSV 提取红色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.config.hsv_lower), np.array(self.config.hsv_upper))

        # 2. 横向膨胀：宽核把同一条带内的碎片连起来，矮核不改变纵向范围
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
        mask = cv2.dilate(mask, h_kernel, iterations=2)

        # 3. 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 4. 按宽高比过滤：宽 >> 高 的才是导航条
        best = None
        best_area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h > h_frame * self.config.max_height_ratio:
                continue
            if w < w_frame * self.config.min_width_ratio:
                continue
            # 优先选面积最大的（最完整的条）
            area = w * h
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

        if best is None:
            return None

        x, y, w, h = best
        pad = self.config.padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_frame, x + w + pad * 2) - x
        h = h + 2 * pad

        logger.debug(f"颜色检测到目标区域: ({x}, {y}, {w}, {h})")
        return (x, y, w, h)
