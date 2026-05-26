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
    hsv_lower: Tuple[int, int, int] = (0, 120, 100)
    hsv_upper: Tuple[int, int, int] = (10, 255, 255)
    min_area: int = 1000
    dilate_kernel_size: int = 5
    padding: int = 10


class ColorDetector:
    """基于 HSV 颜色空间的 ROI 检测器"""

    def __init__(self, config: ColorDetectorConfig):
        self.config = config
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (config.dilate_kernel_size, config.dilate_kernel_size)
        )

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        在帧中检测目标颜色区域。

        参数:
            frame: BGR 图像 (H, W, C)

        返回:
            (x, y, w, h) 或 None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array(self.config.hsv_lower), np.array(self.config.hsv_upper))

        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        valid = [c for c in contours if cv2.contourArea(c) > self.config.min_area]
        if not valid:
            return None

        largest = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        pad = self.config.padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = w + 2 * pad
        h = h + 2 * pad

        logger.debug(f"颜色检测到目标区域: ({x}, {y}, {w}, {h})")
        return (x, y, w, h)
