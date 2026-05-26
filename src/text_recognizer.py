"""
文本识别模块
"""
import numpy as np
from typing import List
import paddle
from paddleocr import PaddleOCR

from .models import BoundingBox, RecognitionResult
from .exceptions import ModelLoadError
from .logger import get_logger

logger = get_logger(__name__)


class TextRecognizer:
    """文本识别器"""
    
    def __init__(
        self, 
        lang: str = "ch", 
        use_gpu: bool = True,
        use_angle_cls: bool = True
    ):
        """
        初始化文本识别器
        
        参数:
            lang: 语言模式 - "ch"(中文), "en"(英文), "ch_en"(中英文混合)
            use_gpu: 是否使用GPU
            use_angle_cls: 是否使用方向分类器
            
        异常:
            ModelLoadError: 模型加载失败
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        
        try:
            logger.info(f"初始化文本识别器 (语言: {lang}, GPU: {use_gpu})")
            
            # 设置 Paddle 设备
            if use_gpu:
                paddle.device.set_device('gpu:0')
                logger.info("设置 Paddle 设备为: gpu:0")
            else:
                paddle.device.set_device('cpu')
                logger.info("设置 Paddle 设备为: cpu")
            
            # 初始化PaddleOCR，使用完整的OCR功能（检测+识别）
            # PaddleOCR 2.x 使用 use_gpu 参数
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu  # PaddleOCR 2.x 使用 use_gpu
            )
            
            logger.info("文本识别器初始化成功")
            
        except Exception as e:
            logger.error(f"文本识别器初始化失败: {str(e)}")
            raise ModelLoadError("TextRecognizer", str(e))
    
    def recognize(
        self, 
        image: np.ndarray, 
        bboxes: List[BoundingBox]
    ) -> List[RecognitionResult]:
        """
        批量识别多个文字区域
        
        参数:
            image: 原始图像 (H, W, C)，BGR格式
            bboxes: 文字区域边界框列表
            
        返回:
            List[RecognitionResult]: 识别结果列表
        """
        if not bboxes:
            logger.info("没有文字区域需要识别")
            return []
        
        results = []
        
        for i, bbox in enumerate(bboxes):
            try:
                # 裁剪文字区域
                text_region = self._crop_region(image, bbox)
                
                # 识别单个区域
                result = self.recognize_single(text_region, bbox)
                
                if result:
                    results.append(result)
                    logger.debug(f"区域 {i+1}/{len(bboxes)}: '{result.text}' (置信度: {result.confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"识别区域 {i+1} 失败: {str(e)}")
                continue
        
        logger.info(f"成功识别 {len(results)}/{len(bboxes)} 个文字区域")
        return results
    
    def recognize_single(
        self, 
        text_region: np.ndarray,
        bbox: BoundingBox = None
    ) -> RecognitionResult:
        """
        识别单个文字区域
        
        参数:
            text_region: 文字区域图像
            bbox: 对应的边界框（可选）
            
        返回:
            RecognitionResult: 识别结果，如果识别失败返回None
        """
        if text_region is None or text_region.size == 0:
            logger.warning("文字区域图像为空")
            return None
        
        try:
            # 调用PaddleOCR识别
            result = self.ocr.ocr(text_region, det=False)
            
            # 解析识别结果
            if result and result[0]:
                # PaddleOCR返回格式: [[[text, confidence]]]
                text, confidence = result[0][0]
                
                # 如果没有提供bbox，创建一个默认的
                if bbox is None:
                    h, w = text_region.shape[:2]
                    bbox = BoundingBox(x=0, y=0, width=w, height=h, confidence=1.0)
                
                recognition_result = RecognitionResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                )
                
                return recognition_result
            else:
                logger.warning("识别结果为空")
                return None
                
        except Exception as e:
            logger.error(f"文字识别失败: {str(e)}")
            return None
    
    def _crop_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        根据边界框裁剪图像区域
        
        参数:
            image: 原始图像
            bbox: 边界框
            
        返回:
            np.ndarray: 裁剪后的图像区域
        """
        h, w = image.shape[:2]
        
        # 确保坐标在图像范围内
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + bbox.height)
        
        # 裁剪区域
        cropped = image[y1:y2, x1:x2]
        
        return cropped
