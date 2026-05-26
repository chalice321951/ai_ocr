# 设计文档：ROI 动态检测与跟踪

## 概述

在现有 OCR 系统中增加 ROI 动态检测与跟踪模块，通过颜色检测定位目标区域 + OpenCV 跟踪器持续跟随，实现视频流/视频/图片中对特定区域的 OCR 识别。

## 系统架构

### 新增模块在整体架构中的位置

```
┌─────────────────────────────────────────────────────────────┐
│                        OCR_System                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Input_Handler                       │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              ROI_Detector (新增)                       │  │
│  │  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │ ColorDetector│  │  Tracker     │                   │  │
│  │  │ (HSV颜色检测)│→│ (CSRT/KCF)   │                   │  │
│  │  └──────────────┘  └──────────────┘                   │  │
│  │  输出: ROI 区域坐标 (x, y, w, h)                      │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          ImageCropper (新增)                           │  │
│  │  根据 ROI 坐标裁剪图像 → ROI 图像                      │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Text_Detector (已有)                       │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Text_Recognizer (已有)                     │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │          OutputManager (新增)                          │  │
│  │  日志归档 / 结果保存 / 可视化                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件设计

### 1. ColorDetector（颜色检测器）

**职责：** 通过 HSV 颜色空间筛选目标区域特征颜色，输出外接矩形坐标。

```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ColorDetectorConfig:
    """颜色检测配置"""
    hsv_lower: Tuple[int, int, int] = (0, 120, 100)   # HSV 下限
    hsv_upper: Tuple[int, int, int] = (10, 255, 255)   # HSV 上限
    min_area: int = 1000          # 最小轮廓面积（过滤噪声）
    dilate_kernel_size: int = 5   # 膨胀核大小
    padding: int = 10             # ROI 外扩像素


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
            (x, y, w, h) 或 None（未检测到）
        """
        # 1. BGR → HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. 颜色范围掩码
        mask = cv2.inRange(hsv, self.config.hsv_lower, self.config.hsv_upper)

        # 3. 形态学操作：膨胀去噪
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        # 4. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 5. 筛选最大轮廓（面积 > min_area）
        valid = [c for c in contours if cv2.contourArea(c) > self.config.min_area]
        if not valid:
            return None

        largest = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # 6. 加 padding
        pad = self.config.padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = w + 2 * pad
        h = h + 2 * pad

        return (x, y, w, h)
```

### 2. ROITracker（ROI 跟踪器）

**职责：** 对颜色检测到的目标区域进行帧间跟踪，定期重新检测修正位置。

```python
import cv2
import numpy as np
from typing import Optional, Tuple


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
        self.current_roi = None  # (x, y, w, h)
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

        # 判断是否需要重新颜色检测
        need_redetect = (
            not self.is_tracking
            or self.frame_count % self.redetect_interval == 0
        )

        if need_redetect:
            roi = self.color_detector.detect(frame)
            if roi is not None:
                # 用检测结果重新初始化跟踪器
                self.tracker = self._create_tracker()
                x, y, w, h = roi
                self.tracker.init(frame, (x, y, w, h))
                self.current_roi = roi
                self.is_tracking = True
                return roi
            elif self.is_tracking:
                # 检测失败但还在跟踪，继续用跟踪器
                pass
            else:
                self.current_roi = None
                return None

        # 使用跟踪器更新
        if self.is_tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                self.current_roi = (x, y, w, h)
                return self.current_roi
            else:
                # 跟踪失败，重置状态
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
```

### 3. ImageCropper（图像裁剪器）

**职责：** 根据 ROI 坐标裁剪图像。

```python
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

        # 边界检查
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)

        return frame[y1:y2, x1:x2].copy()
```

### 4. OutputManager（输出管理器）

**职责：** 管理日志目录和结果输出目录。

```python
import os
import json
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List


class OutputManager:
    """输出目录与日志管理"""

    def __init__(self, base_dir: str = ".", log_level: str = "INFO"):
        self.base_dir = base_dir
        self.log_dir = os.path.join(base_dir, "logs")
        self.res_dir = os.path.join(base_dir, "res")
        self.session_dir = None   # 当次运行的结果目录
        self.session_prefix = None

        os.makedirs(self.log_dir, exist_ok=True)

    def start_session(self, detect_type: str = "roi_detect"):
        """
        创建本次运行的输出目录。

        参数:
            detect_type: 检测类别，如 "roi_detect" / "full_detect"
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_prefix = f"{ts}_{detect_type}"
        self.session_dir = os.path.join(self.res_dir, self.session_prefix)
        os.makedirs(self.session_dir, exist_ok=True)

        roi_crop_dir = os.path.join(self.session_dir, "roi_crop")
        os.makedirs(roi_crop_dir, exist_ok=True)

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

        # 追加模式：如果文件已存在则读取并追加
        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        existing.append(result)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def get_log_path(self) -> str:
        """获取当日日志文件路径"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{date_str}.log")
```

### 5. 配置扩展

在 `config.yaml` 中新增 ROI 相关配置：

```yaml
# ROI 检测配置
roi:
  enabled: false            # 是否启用 ROI 模式

  # 颜色检测参数
  color_detect:
    hsv_lower: [0, 120, 100]     # HSV 下限 [H, S, V]
    hsv_upper: [10, 255, 255]    # HSV 上限 [H, S, V]
    min_area: 1000               # 最小轮廓面积
    dilate_kernel_size: 5        # 膨胀核大小
    padding: 10                  # ROI 外扩像素

  # 跟踪器参数
  tracker:
    type: "CSRT"                 # 跟踪器类型: CSRT, KCF
    redetect_interval: 30        # 每 N 帧重新颜色检测

# 输出配置
output:
  save_result: true
  visualize: true
  save_roi_crop: true            # 是否保存 ROI 裁剪图

# 日志配置
logging:
  level: "INFO"
  log_file: "logs/ocr_system.log"
```

### 6. Config 类扩展

```python
# 在 src/config.py 中新增
@dataclass
class ColorDetectConfig:
    hsv_lower: tuple = (0, 120, 100)
    hsv_upper: tuple = (10, 255, 255)
    min_area: int = 1000
    dilate_kernel_size: int = 5
    padding: int = 10


@dataclass
class TrackerConfig:
    type: str = "CSRT"
    redetect_interval: int = 30


@dataclass
class ROIConfig:
    enabled: bool = False
    color_detect: ColorDetectConfig = field(default_factory=ColorDetectConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
```

## 数据流设计

### 单张图片 + ROI 模式

```
1. InputHandler.load_image(path) → frame
2. ColorDetector.detect(frame) → roi (x, y, w, h)
3. IF roi is None → 输出空结果，结束
4. ImageCropper.crop(frame, roi) → roi_image
5. TextDetector.detect(roi_image) → List[BoundingBox]
6. TextRecognizer.recognize(roi_image, bboxes) → List[RecognitionResult]
7. OutputManager.save_annotated_image(...) / save_ocr_result(...)
```

### 视频/流 + ROI 模式

```
1. InputHandler.open_stream(source) → iterator(frame)
2. ROITracker.reset()
3. FOR EACH frame:
   a. ROITracker.update(frame) → roi
   b. IF roi is None → 跳过 OCR，记录日志
   c. ImageCropper.crop(frame, roi) → roi_image
   d. TextDetector.detect(roi_image) → List[BoundingBox]
   e. TextRecognizer.recognize(roi_image, bboxes) → List[RecognitionResult]
   f. 可视化：在 frame 上画出 ROI 框 + OCR 文字
   g. OutputManager.save_annotated_image(frame, f"annotated_{idx:04d}.jpg")
   h. OutputManager.save_ocr_result(result)
```

### 与现有系统的集成点

```
OCRSystem.process_stream(stream_source):
    if config.roi.enabled:
        # ROI 模式
        tracker = ROITracker(color_detector, config.roi.tracker)
        cropper = ImageCropper()
        for frame in stream:
            roi = tracker.update(frame)
            if roi:
                roi_img = cropper.crop(frame, roi)
                result = self.text_recognize(roi_img)  # 复用现有识别
            # ... 输出
    else:
        # 全图模式（现有逻辑不变）
        for frame in stream:
            bboxes = self.text_detector.detect(frame)
            result = self.text_recognize(frame, bboxes)
```

## 文件结构变化

```
src/
├── color_detector.py      # 新增：颜色检测器
├── roi_tracker.py         # 新增：ROI 跟踪器
├── image_cropper.py       # 新增：图像裁剪器
├── output_manager.py      # 新增：输出管理器
├── config.py              # 修改：新增 ROIConfig
├── ocr_system.py          # 修改：集成 ROI 流程
├── text_detector.py       # 不变
├── text_recognizer.py     # 不变
├── models.py              # 不变
├── input_handler.py       # 不变
├── result_formatter.py    # 不变
└── ...

config/
└── config.yaml            # 修改：新增 roi 和 output 配置段

logs/                      # 新增目录
└── 2026-05-26.log

res/                       # 新增目录
└── 20260526_153012_roi_detect/
    ├── annotated_0001.jpg
    ├── ocr_result.json
    └── roi_crop/
        └── roi_0001.jpg
```

## 需求追溯矩阵

| 组件 | 覆盖需求 |
|------|---------|
| ColorDetector | F1.1, F1.2, F1.3, F1.4, F1.5 |
| ROITracker | F2.1, F2.2, F2.3, F2.4, F2.5 |
| ImageCropper | F3.1, F3.4 |
| OCRSystem 集成 | F3.2, F3.3, F4.1, F4.2, F4.3, F4.4 |
| Config / config.yaml | F5.1, F5.2, F5.3, F5.4 |
| 异常处理 | F6.1, F6.2, F6.3, F6.4 |
| OutputManager | F7.1, F7.2, F7.3, F7.4, F7.5, F7.6 |
