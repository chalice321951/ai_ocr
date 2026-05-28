"""
结果格式化模块
"""
import json
import csv
import io
import os
import platform
import cv2
import numpy as np
from typing import List
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from PIL import Image, ImageDraw, ImageFont

from .models import RecognitionResult, OCRResult
from .logger import get_logger

logger = get_logger(__name__)


def get_chinese_font(font_size: int = 20) -> ImageFont.FreeTypeFont:
    """
    获取支持中文的字体

    参数:
        font_size: 字体大小

    返回:
        ImageFont.FreeTypeFont: 字体对象
    """
    # 常见的中文字体路径
    font_paths = []

    # 根据操作系统选择字体路径
    system = platform.system()
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/msyh.ttc",         # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",       # 宋体
            "C:/Windows/Fonts/STKAITI.TTF",      # 华文楷体
        ]
    elif system == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", # 文泉驿微米黑
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ]

    # 尝试加载字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                logger.warning(f"加载字体失败 {font_path}: {e}")
                continue

    # 如果没有找到字体，使用默认字体
    logger.warning("未找到中文字体，使用默认字体（中文可能显示为方框）")
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except:
        return ImageFont.load_default()


class ResultFormatter:
    """结果格式化器"""

    @staticmethod
    def to_xml(results: List[RecognitionResult]) -> str:
        """
        转换为XML格式
        
        参数:
            results: 识别结果列表
            
        返回:
            str: XML字符串
        """
        root = Element('ocr_results')
        root.set('total_count', str(len(results)))
        
        for i, result in enumerate(results):
            result_elem = SubElement(root, 'result')
            result_elem.set('id', str(i))
            
            # 添加文字内容
            text_elem = SubElement(result_elem, 'text')
            text_elem.text = result.text
            
            # 添加置信度
            confidence_elem = SubElement(result_elem, 'confidence')
            confidence_elem.text = str(result.confidence)
            
            # 添加边界框
            bbox_elem = SubElement(result_elem, 'bounding_box')
            bbox_elem.set('x', str(result.bbox.x))
            bbox_elem.set('y', str(result.bbox.y))
            bbox_elem.set('width', str(result.bbox.width))
            bbox_elem.set('height', str(result.bbox.height))
        
        # 格式化XML
        xml_str = tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent='  ')
    
    @staticmethod
    def to_csv(results: List[RecognitionResult]) -> str:
        """
        转换为CSV格式
        
        参数:
            results: 识别结果列表
            
        返回:
            str: CSV字符串
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        writer.writerow(['id', 'text', 'confidence', 'x', 'y', 'width', 'height'])
        
        # 写入数据
        for i, result in enumerate(results):
            writer.writerow([
                i,
                result.text,
                result.confidence,
                result.bbox.x,
                result.bbox.y,
                result.bbox.width,
                result.bbox.height
            ])
        
        return output.getvalue()
    
    @staticmethod
    def visualize(
        image: np.ndarray,
        results: List[RecognitionResult],
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        在图像上可视化结果（支持中文）

        参数:
            image: 原始图像
            results: 识别结果
            confidence_threshold: 置信度阈值，用于区分颜色

        返回:
            np.ndarray: 标注后的图像
        """
        # 将OpenCV图像转换为PIL图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 获取中文字体
        font_size = 20
        font = get_chinese_font(font_size)

        for result in results:
            bbox = result.bbox
            text = result.text
            confidence = result.confidence

            # 根据置信度选择颜色
            if confidence >= confidence_threshold:
                color = (0, 255, 0)  # 绿色 - 高置信度
                text_bg_color = (0, 200, 0)
            else:
                color = (255, 165, 0)  # 橙色 - 低置信度
                text_bg_color = (200, 130, 0)

            # 绘制边界框
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 准备文字标签
            label = f"{text} ({confidence:.2f})"

            # 计算文字大小
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # 计算文字位置（在框上方或下方）
            if y1 - text_height - 10 > 0:
                text_y = y1 - text_height - 5
            else:
                text_y = y2 + 5

            # 绘制文字背景
            draw.rectangle(
                [x1, text_y, x1 + text_width + 4, text_y + text_height + 4],
                fill=text_bg_color
            )

            # 绘制文字
            draw.text((x1 + 2, text_y + 2), label, fill=(255, 255, 255), font=font)

        # 将PIL图像转换回OpenCV格式
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        logger.info(f"可视化完成: 标注了 {len(results)} 个识别结果")
        return result_image
    
    @staticmethod
    def format_ocr_result(ocr_result: OCRResult, format_type: str = 'json') -> str:
        """
        格式化OCRResult对象
        
        参数:
            ocr_result: OCR结果对象
            format_type: 格式类型 ('json', 'xml', 'csv')
            
        返回:
            str: 格式化后的字符串
        """
        if format_type == 'json':
            return json.dumps(ocr_result.to_dict(), ensure_ascii=False, indent=2)
        elif format_type == 'xml':
            return ResultFormatter.to_xml(ocr_result.results)
        elif format_type == 'csv':
            return ResultFormatter.to_csv(ocr_result.results)
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
