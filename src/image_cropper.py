"""
ROI 图像裁剪器
"""
import numpy as np
from typing import Tuple


class ImageCropper:
    """ROI 图像裁剪"""

    @staticmethod
    def crop(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        裁剪 ROI 区域。

        参数:
            frame: 原始图像
            roi: (x, y, w, h)

        返回:
            裁剪后的图像
        """
        x, y, w, h = roi
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        return frame[y1:y2, x1:x2].copy()
