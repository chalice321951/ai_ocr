"""
核心数据模型定义
"""
import time
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    """文字区域边界框"""
    x: int  # 左上角x坐标
    y: int  # 左上角y坐标
    width: int  # 宽度
    height: int  # 高度
    confidence: float  # 检测置信度 (0-1)
    
    def __post_init__(self):
        """验证数据有效性"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在0-1之间: {self.confidence}")
        if self.width < 0 or self.height < 0:
            raise ValueError(f"宽度和高度必须非负: width={self.width}, height={self.height}")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence
        }
    
    def get_corners(self) -> List[tuple]:
        """
        获取边界框的四个角点坐标
        
        返回:
            List[tuple]: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        return [
            (self.x, self.y),  # 左上
            (self.x + self.width, self.y),  # 右上
            (self.x + self.width, self.y + self.height),  # 右下
            (self.x, self.y + self.height)  # 左下
        ]


@dataclass
class RecognitionResult:
    """文字识别结果"""
    text: str  # 识别的文字内容
    confidence: float  # 识别置信度 (0-1)
    bbox: BoundingBox  # 对应的边界框
    
    def __post_init__(self):
        """验证数据有效性"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在0-1之间: {self.confidence}")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox.to_dict()
        }


@dataclass
class OCRResult:
    """完整的OCR结果"""
    results: List[RecognitionResult]  # 识别结果列表
    processing_time: float  # 处理时间（秒）
    timestamp: float = field(default_factory=time.time)  # 时间戳
    image_path: Optional[str] = None  # 图像路径（可选）
    
    def __post_init__(self):
        """验证数据有效性"""
        if self.processing_time < 0:
            raise ValueError(f"处理时间必须非负: {self.processing_time}")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'results': [r.to_dict() for r in self.results],
            'processing_time': self.processing_time,
            'timestamp': self.timestamp,
            'image_path': self.image_path,
            'total_texts': len(self.results)
        }
    
    def get_all_texts(self) -> List[str]:
        """
        获取所有识别的文字
        
        返回:
            List[str]: 文字列表
        """
        return [r.text for r in self.results]
    
    def filter_by_confidence(self, threshold: float) -> List[RecognitionResult]:
        """
        根据置信度过滤结果
        
        参数:
            threshold: 置信度阈值
            
        返回:
            List[RecognitionResult]: 过滤后的结果列表
        """
        return [r for r in self.results if r.confidence >= threshold]


@dataclass
class AccuracyMetrics:
    """精度评估指标"""
    accuracy: float = 0.0  # 准确率
    precision: float = 0.0  # 精确率
    recall: float = 0.0  # 召回率
    f1_score: float = 0.0  # F1分数
    char_accuracy: float = 0.0  # 字符级准确率
    word_accuracy: float = 0.0  # 单词级准确率
    total_samples: int = 0  # 总样本数
    correct_samples: int = 0  # 正确样本数
    
    def __post_init__(self):
        """验证数据有效性"""
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 
                           'char_accuracy', 'word_accuracy']:
            value = getattr(self, metric_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{metric_name}必须在0-1之间: {value}")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'char_accuracy': self.char_accuracy,
            'word_accuracy': self.word_accuracy,
            'total_samples': self.total_samples,
            'correct_samples': self.correct_samples
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"AccuracyMetrics(\n"
            f"  准确率: {self.accuracy:.2%}\n"
            f"  精确率: {self.precision:.2%}\n"
            f"  召回率: {self.recall:.2%}\n"
            f"  F1分数: {self.f1_score:.2%}\n"
            f"  字符级准确率: {self.char_accuracy:.2%}\n"
            f"  单词级准确率: {self.word_accuracy:.2%}\n"
            f"  总样本数: {self.total_samples}\n"
            f"  正确样本数: {self.correct_samples}\n"
            f")"
        )
