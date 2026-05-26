"""
文本检测模块
"""
import numpy as np
from typing import List
import paddle
from paddleocr import PaddleOCR

from .models import BoundingBox
from .exceptions import ModelLoadError
from .logger import get_logger

logger = get_logger(__name__)


class TextDetector:
    """文本检测器"""
    
    def __init__(self, use_gpu: bool = True, use_angle_cls: bool = True):
        """
        初始化文本检测器
        
        参数:
            use_gpu: 是否使用GPU
            use_angle_cls: 是否使用方向分类器
            
        异常:
            ModelLoadError: 模型加载失败
        """
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        
        try:
            logger.info(f"初始化文本检测器 (GPU: {use_gpu}, 方向分类: {use_angle_cls})")
            
            # 设置 Paddle 设备
            if use_gpu:
                paddle.device.set_device('gpu:0')
                logger.info("设置 Paddle 设备为: gpu:0")
            else:
                paddle.device.set_device('cpu')
                logger.info("设置 Paddle 设备为: cpu")
            
            # 初始化PaddleOCR，只使用检测功能
            # PaddleOCR 3.x API 变化：不再使用 use_gpu 参数，通过设备设置控制
            try:
                # PaddleOCR 3.x 新 API
                self.ocr = PaddleOCR(
                    use_angle_cls=use_angle_cls,
                    lang='ch'
                )
            except TypeError:
                # 兼容 PaddleOCR 2.x 旧 API
                self.ocr = PaddleOCR(
                    use_angle_cls=use_angle_cls,
                    lang='ch',
                    use_gpu=use_gpu
                )
            
            logger.info("文本检测器初始化成功")
            
        except Exception as e:
            logger.error(f"文本检测器初始化失败: {str(e)}")
            raise ModelLoadError("TextDetector", str(e))
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        检测图像中的文字区域
        
        参数:
            image: 输入图像数组 (H, W, C)，BGR格式
            
        返回:
            List[BoundingBox]: 文字区域边界框列表，空列表表示未检测到文字
        """
        if image is None or image.size == 0:
            logger.warning("输入图像为空")
            return []
        
        try:
            logger.debug(f"开始检测文字区域，图像尺寸: {image.shape}")
            
            # 调用PaddleOCR检测（rec=False表示只检测不识别）
            result = self.ocr.ocr(image, rec=False)

            # 调试：打印result的类型和结构
            logger.debug(f"PaddleOCR返回结果类型: {type(result)}")
            if result is not None and len(result) > 0:
                logger.debug(f"result[0]类型: {type(result[0])}, 内容: {result[0]}")

            # 解析检测结果
            bboxes = []

            if result is not None and len(result) > 0 and result[0] is not None and len(result[0]) > 0:
                for detection in result[0]:
                    # detection是一个包含坐标点的列表
                    # 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    points = detection
                    
                    # 计算边界框
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    x_min = int(min(x_coords))
                    y_min = int(min(y_coords))
                    x_max = int(max(x_coords))
                    y_max = int(max(y_coords))
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # 检测置信度设为1.0（PaddleOCR检测阶段不返回置信度）
                    confidence = 1.0
                    
                    bbox = BoundingBox(
                        x=x_min,
                        y=y_min,
                        width=width,
                        height=height,
                        confidence=confidence
                    )
                    bboxes.append(bbox)
                
                logger.info(f"检测到 {len(bboxes)} 个文字区域")
            else:
                logger.info("未检测到文字区域")
            
            return bboxes
            
        except Exception as e:
            import traceback
            logger.error(f"文字检测失败: {str(e)}")
            logger.error(f"详细堆栈:\n{traceback.format_exc()}")
            # 返回空列表而不是抛出异常，保持系统鲁棒性
            return []
