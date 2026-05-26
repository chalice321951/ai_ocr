"""
ROI 跟踪器 - 颜色检测 + OpenCV 跟踪器组合
"""
import cv2
import numpy as np
from typing import Optional, Tuple

from .color_detector import ColorDetector
from .logger import get_logger

logger = get_logger(__name__)


class ROITracker:
    """颜色检测 + OpenCV 跟踪器组合"""

    def __init__(self, color_detector: ColorDetector,
                 tracker_type: str = "CSRT",
                 redetect_interval: int = 30):
        """
        参数:
            color_detector: 颜色检测器
            tracker_type: 跟踪器类型 "CSRT" 或 "KCF"
            redetect_interval: 每 N 帧重新颜色检测
        """
        self.color_detector = color_detector
        self.tracker_type = tracker_type
        self.redetect_interval = redetect_interval

        self.tracker = None
        self.frame_count = 0
        self.current_roi = None
        self.is_tracking = False

    def _create_tracker(self):
        """创建 OpenCV 跟踪器"""
        if self.tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        else:
            raise ValueError(f"不支持的跟踪器类型: {self.tracker_type}")

    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        更新跟踪状态，返回当前 ROI。

        参数:
            frame: BGR 图像

        返回:
            (x, y, w, h) 或 None
        """
        self.frame_count += 1

        need_redetect = (
            not self.is_tracking
            or self.frame_count % self.redetect_interval == 0
        )

        if need_redetect:
            roi = self.color_detector.detect(frame)
            if roi is not None:
                self.tracker = self._create_tracker()
                x, y, w, h = roi
                self.tracker.init(frame, (x, y, w, h))
                self.current_roi = roi
                self.is_tracking = True
                logger.debug(f"颜色检测重新定位 ROI: {roi}")
                return roi
            elif self.is_tracking:
                logger.debug("颜色检测失败，继续使用跟踪器")
            else:
                self.current_roi = None
                return None

        if self.is_tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                self.current_roi = (x, y, w, h)
                return self.current_roi
            else:
                logger.warning("跟踪器丢失目标，重置状态")
                self.is_tracking = False
                self.tracker = None
                self.current_roi = None
                return None

        return self.current_roi

    def reset(self):
        """重置跟踪器状态"""
        self.tracker = None
        self.frame_count = 0
        self.current_roi = None
        self.is_tracking = False
