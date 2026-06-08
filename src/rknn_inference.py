"""
RKNN推理引擎 - RK3576 NPU加速
用于替代PaddleOCR，在RK3576上使用NPU进行推理
"""
import cv2
import numpy as np
from typing import List, Tuple
from rknnlite.api import RKNNLite


class RKNNDetector:
    """RKNN文本检测器"""

    def __init__(self, model_path: str, core_mask: int = 0):
        """
        初始化检测器

        Args:
            model_path: RKNN模型路径
            core_mask: NPU核心掩码 (0=自动, 1=core0, 2=core1, 3=both)
        """
        self.model_path = model_path
        self.core_mask = core_mask
        self.input_size = (640, 640)  # 检测模型输入尺寸

        # 加载模型
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载RKNN模型失败: {model_path}")

        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"初始化RKNN运行时失败")

        print(f"✓ 检测模型加载成功: {model_path}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        预处理图像 - PaddleOCR检测模型专用

        Args:
            image: 输入图像 (H, W, C) BGR格式

        Returns:
            preprocessed: 预处理后的图像 (1, 3, 640, 640)
            ratio_h: 高度缩放比例
            ratio_w: 宽度缩放比例
        """
        h, w = image.shape[:2]

        # 计算缩放比例
        ratio_h = self.input_size[0] / h
        ratio_w = self.input_size[1] / w

        # Resize
        resized = cv2.resize(image, self.input_size)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 归一化到 [-1, 1] (PaddleOCR使用的归一化方式)
        normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5

        # HWC to CHW
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched, ratio_h, ratio_w

    def postprocess(self, output: np.ndarray, ratio_h: float, ratio_w: float,
                   thresh: float = 0.3, box_thresh: float = 0.6) -> List[np.ndarray]:
        """
        后处理 - DB算法

        Args:
            output: 模型输出 (1, 1, H, W)
            ratio_h: 高度缩放比例
            ratio_w: 宽度缩放比例
            thresh: 二值化阈值
            box_thresh: 框置信度阈值

        Returns:
            检测框列表 [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
        """
        # 获取概率图
        pred = output[0, 0, :, :]  # (H, W)

        # 二值化
        bitmap = (pred > thresh).astype(np.uint8)

        # 形态学操作 - 膨胀
        kernel = np.ones((3, 3), np.uint8)
        bitmap = cv2.dilate(bitmap, kernel, iterations=2)

        # 查找轮廓
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

            # 计算框的面积
            area = cv2.contourArea(box)
            if area < 10:  # 过滤太小的框
                continue

            # 计算框内的平均得分
            mask = np.zeros_like(pred, dtype=np.uint8)
            cv2.fillPoly(mask, [box.astype(np.int32)], 1)
            score = (pred * mask).sum() / (mask.sum() + 1e-6)

            if score < box_thresh:
                continue

            # 映射回原图坐标
            box[:, 0] = box[:, 0] / ratio_w
            box[:, 1] = box[:, 1] / ratio_h

            boxes.append(box)

        return boxes

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """
        检测图像中的文字区域

        Args:
            image: 输入图像 (H, W, C) BGR格式

        Returns:
            检测框列表 [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
        """
        # 预处理
        input_data, ratio_h, ratio_w = self.preprocess(image)

        # NPU推理
        outputs = self.rknn.inference(inputs=[input_data])

        # 后处理
        boxes = self.postprocess(outputs[0], ratio_h, ratio_w)

        return boxes

    def __del__(self):
        """释放资源"""
        if hasattr(self, 'rknn'):
            self.rknn.release()


class RKNNRecognizer:
    """RKNN文本识别器"""

    def __init__(self, model_path: str, dict_path: str, core_mask: int = 0):
        """
        初始化识别器

        Args:
            model_path: RKNN模型路径
            dict_path: 字符字典路径
            core_mask: NPU核心掩码
        """
        self.model_path = model_path
        self.core_mask = core_mask
        self.input_size = (320, 48)  # 识别模型输入尺寸 (W, H)

        # 加载字符字典
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.char_dict = ['blank'] + [line.strip() for line in f]

        # 加载模型
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载RKNN模型失败: {model_path}")

        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"初始化RKNN运行时失败")

        print(f"✓ 识别模型加载成功: {model_path}")
        print(f"  字符数: {len(self.char_dict)}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像 - PaddleOCR识别模型专用

        Args:
            image: 输入图像 (H, W, C) BGR格式

        Returns:
            preprocessed: 预处理后的图像 (1, 3, 48, 320)
        """
        h, w = image.shape[:2]

        # 计算缩放比例（保持宽高比）
        ratio = self.input_size[1] / h
        new_w = int(w * ratio)

        # Resize
        resized = cv2.resize(image, (new_w, self.input_size[1]))

        # 如果宽度超过320，裁剪；如果不足，填充
        if new_w > self.input_size[0]:
            resized = resized[:, :self.input_size[0], :]
        elif new_w < self.input_size[0]:
            # 右侧填充白色
            pad_width = self.input_size[0] - new_w
            resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_width,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 归一化到 [-1, 1] (PaddleOCR使用的归一化方式)
        normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5

        # HWC to CHW
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def decode_ctc(self, preds: np.ndarray) -> Tuple[str, float]:
        """
        CTC解码

        Args:
            preds: 模型输出 (T, C)

        Returns:
            text: 识别文字
            confidence: 置信度
        """
        # Argmax
        indices = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)

        # CTC去重
        chars = []
        last_idx = 0  # blank
        for idx, conf in zip(indices, confidences):
            if idx != 0 and idx != last_idx:  # 不是blank且与上一个不同
                chars.append(self.char_dict[idx])
            last_idx = idx

        text = ''.join(chars)
        confidence = float(np.mean(confidences))

        return text, confidence

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """
        识别文字区域

        Args:
            image: 输入图像 (H, W, C) BGR格式

        Returns:
            text: 识别文字
            confidence: 置信度
        """
        # 预处理
        input_data = self.preprocess(image)

        # NPU推理
        outputs = self.rknn.inference(inputs=[input_data])

        # CTC解码
        text, confidence = self.decode_ctc(outputs[0][0])

        return text, confidence

    def __del__(self):
        """释放资源"""
        if hasattr(self, 'rknn'):
            self.rknn.release()
