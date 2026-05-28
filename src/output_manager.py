"""
输出目录与日志管理
"""
import os
import json
import cv2
import numpy as np
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


class OutputManager:
    """输出目录与日志管理"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.log_dir = os.path.join(base_dir, "logs")
        self.res_dir = os.path.join(base_dir, "res")
        self.session_dir = None

        os.makedirs(self.log_dir, exist_ok=True)

    def start_session(self, detect_type: str = "roi_detect") -> str:
        """
        创建本次运行的输出目录。

        参数:
            detect_type: 检测类别，如 "roi_detect" / "full_detect"

        返回:
            本次运行的输出目录路径
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{ts}_{detect_type}"
        self.session_dir = os.path.join(self.res_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        logger.info(f"输出目录: {self.session_dir}")
        return self.session_dir

    def save_annotated_image(self, frame: np.ndarray, filename: str):
        """保存可视化标注图"""
        if self.session_dir is None:
            return
        path = os.path.join(self.session_dir, filename)
        cv2.imwrite(path, frame)

    def save_roi_crop(self, roi_image: np.ndarray, filename: str):
        """保存 ROI 裁剪图"""
        if self.session_dir is None:
            return
        path = os.path.join(self.session_dir, "roi_crop", filename)
        cv2.imwrite(path, roi_image)

    def save_ocr_result(self, result: dict, filename: str = "ocr_result.json"):
        """保存 OCR 结果 JSON"""
        if self.session_dir is None:
            return
        path = os.path.join(self.session_dir, filename)

        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        existing.append(result)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
