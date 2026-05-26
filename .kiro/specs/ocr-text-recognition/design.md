# 设计文档：OCR文字识别系统

## 概述

本文档描述OCR文字识别系统的技术设计。系统采用两阶段架构：文本检测（Text Detection）和文本识别（Text Recognition），使用PaddleOCR作为核心框架，支持从图片、视频和数据流中提取中英文文字。

## 系统架构

### 高层架构

```
┌─────────────────────────────────────────────────────────┐
│                    OCR_System                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Input_Handler                             │  │
│  │  - ImageLoader                                    │  │
│  │  - VideoProcessor                                 │  │
│  │  - StreamProcessor                                │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                       │
│                  ▼                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Text_Detector                             │  │
│  │  - PaddleOCR Detection Model                      │  │
│  │  - Returns: List[BoundingBox]                     │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                       │
│                  ▼                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Text_Recognizer                           │  │
│  │  - PaddleOCR Recognition Model                    │  │
│  │  - Returns: List[RecognitionResult]               │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                       │
│                  ▼                                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Result_Formatter                          │  │
│  │  - JSON/XML/CSV output                            │  │
│  │  - Visualization                                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 核心组件设计

### 1. OCR_System（主系统类）

**职责：** 协调所有子组件，提供统一的OCR接口

**接口：**

```python
class OCRSystem:
    def __init__(self, config: Config):
        """
        初始化OCR系统
        
        参数:
            config: 配置对象，包含模型路径、设备选择等
        """
        pass
    
    def process_image(self, image_path: str) -> OCRResult:
        """
        处理单张图片
        
        参数:
            image_path: 图片文件路径
            
        返回:
            OCRResult: 包含检测框和识别文字的结果对象
            
        需求: 1.1, 2.1, 3.1, 4.1
        """
        pass
    
    def process_video(self, video_path: str, frame_interval: int = 1) -> List[OCRResult]:
        """
        处理视频文件
        
        参数:
            video_path: 视频文件路径
            frame_interval: 帧间隔，每隔多少帧处理一次
            
        返回:
            List[OCRResult]: 每一帧的OCR结果列表
            
        需求: 1.2, 4.2, 7.4
        """
        pass
    
    def process_stream(self, stream_source) -> Iterator[OCRResult]:
        """
        处理数据流
        
        参数:
            stream_source: 数据流源
            
        返回:
            Iterator[OCRResult]: OCR结果迭代器
            
        需求: 1.3
        """
        pass
    
    def evaluate_accuracy(self, test_dataset: Dataset) -> AccuracyMetrics:
        """
        评估模型精度
        
        参数:
            test_dataset: 测试数据集
            
        返回:
            AccuracyMetrics: 包含准确率、召回率、F1分数的指标对象
            
        需求: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        pass
```

### 2. Input_Handler（输入处理模块）

**职责：** 处理不同类型的输入源，统一转换为图像数组

**接口：**

```python
class ImageLoader:
    def load(self, image_path: str) -> np.ndarray:
        """
        加载图片文件
        
        参数:
            image_path: 图片路径
            
        返回:
            np.ndarray: 图像数组 (H, W, C)
            
        异常:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
            
        需求: 1.1, 1.4, 8.1, 8.2
        """
        pass

class VideoProcessor:
    def extract_frames(self, video_path: str, frame_interval: int = 1) -> Iterator[np.ndarray]:
        """
        从视频中提取帧
        
        参数:
            video_path: 视频路径
            frame_interval: 帧间隔
            
        返回:
            Iterator[np.ndarray]: 帧图像迭代器
            
        异常:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的视频格式
            
        需求: 1.2, 1.5, 8.1, 8.2
        """
        pass

class StreamProcessor:
    def process_stream(self, stream_source) -> Iterator[np.ndarray]:
        """
        处理数据流
        
        参数:
            stream_source: 数据流源
            
        返回:
            Iterator[np.ndarray]: 图像帧迭代器
            
        需求: 1.3
        """
        pass
```

### 3. Text_Detector（文本检测模块）

**职责：** 检测图像中的文字区域

**接口：**

```python
class TextDetector:
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        初始化文本检测器
        
        参数:
            model_path: 检测模型路径
            use_gpu: 是否使用GPU
            
        异常:
            ModelLoadError: 模型加载失败
            
        需求: 9.3, 9.4, 8.3
        """
        pass
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        检测图像中的文字区域
        
        参数:
            image: 输入图像数组
            
        返回:
            List[BoundingBox]: 文字区域边界框列表
            空列表表示未检测到文字
            
        需求: 2.1, 2.2, 2.4, 2.5
        """
        pass

class BoundingBox:
    """文字区域边界框"""
    def __init__(self, x: int, y: int, width: int, height: int, confidence: float):
        """
        参数:
            x: 左上角x坐标
            y: 左上角y坐标
            width: 宽度
            height: 高度
            confidence: 检测置信度
            
        需求: 2.5
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
```

### 4. Text_Recognizer（文本识别模块）

**职责：** 从文字区域中识别文字内容

**接口：**

```python
class TextRecognizer:
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        初始化文本识别器
        
        参数:
            model_path: 识别模型路径
            use_gpu: 是否使用GPU
            
        异常:
            ModelLoadError: 模型加载失败
            
        需求: 9.3, 9.4, 8.3
        """
        pass
    
    def recognize(self, image: np.ndarray, bboxes: List[BoundingBox]) -> List[RecognitionResult]:
        """
        识别文字区域中的文字
        
        参数:
            image: 原始图像
            bboxes: 文字区域边界框列表
            
        返回:
            List[RecognitionResult]: 识别结果列表
            
        需求: 3.1, 3.2, 3.3, 3.4, 3.6, 4.2
        """
        pass
    
    def recognize_single(self, text_region: np.ndarray) -> RecognitionResult:
        """
        识别单个文字区域
        
        参数:
            text_region: 文字区域图像
            
        返回:
            RecognitionResult: 识别结果
            
        需求: 3.1, 3.6
        """
        pass

class RecognitionResult:
    """文字识别结果"""
    def __init__(self, text: str, confidence: float, bbox: BoundingBox):
        """
        参数:
            text: 识别的文字内容
            confidence: 识别置信度
            bbox: 对应的边界框
            
        需求: 3.6, 4.3
        """
        self.text = text
        self.confidence = confidence
        self.bbox = bbox
```

### 5. Result_Formatter（结果格式化模块）

**职责：** 格式化和可视化OCR结果

**接口：**

```python
class ResultFormatter:
    def to_json(self, results: List[RecognitionResult]) -> str:
        """
        转换为JSON格式
        
        参数:
            results: 识别结果列表
            
        返回:
            str: JSON字符串
            
        需求: 7.1, 7.2
        """
        pass
    
    def to_xml(self, results: List[RecognitionResult]) -> str:
        """
        转换为XML格式
        
        需求: 10.5
        """
        pass
    
    def to_csv(self, results: List[RecognitionResult]) -> str:
        """
        转换为CSV格式
        
        需求: 10.5
        """
        pass
    
    def visualize(self, image: np.ndarray, results: List[RecognitionResult]) -> np.ndarray:
        """
        在图像上可视化结果
        
        参数:
            image: 原始图像
            results: 识别结果
            
        返回:
            np.ndarray: 标注后的图像
            
        需求: 7.5
        """
        pass

class OCRResult:
    """完整的OCR结果"""
    def __init__(self, results: List[RecognitionResult], processing_time: float):
        """
        参数:
            results: 识别结果列表
            processing_time: 处理时间（秒）
            
        需求: 4.4, 7.2
        """
        self.results = results
        self.processing_time = processing_time
        self.timestamp = time.time()
```

### 6. Config（配置管理）

**职责：** 管理系统配置

**接口：**

```python
class Config:
    """系统配置类"""
    def __init__(self, config_path: str = None):
        """
        从配置文件加载配置
        
        参数:
            config_path: 配置文件路径，None则使用默认配置
            
        需求: 10.1, 10.2, 10.3, 10.5
        """
        self.det_model_type = "DB"  # 检测模型类型
        self.rec_model_type = "CRNN"  # 识别模型类型
        self.use_gpu = True  # 是否使用GPU
        self.confidence_threshold = 0.5  # 置信度阈值
        self.output_format = "json"  # 输出格式
        self.lang = "ch"  # 语言：ch（中文）, en（英文）, ch_en（中英文）
        
    def load_from_file(self, config_path: str):
        """从文件加载配置"""
        pass
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        pass
```

### 7. Accuracy_Evaluator（精度评估模块）

**职责：** 评估模型精度

**接口：**

```python
class AccuracyEvaluator:
    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> AccuracyMetrics:
        """
        评估识别精度
        
        参数:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            
        返回:
            AccuracyMetrics: 精度指标
            
        需求: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        pass
    
    def character_accuracy(self, pred: str, gt: str) -> float:
        """计算字符级准确率"""
        pass
    
    def word_accuracy(self, pred: str, gt: str) -> float:
        """计算单词级准确率"""
        pass

class AccuracyMetrics:
    """精度指标"""
    def __init__(self):
        self.accuracy = 0.0  # 准确率
        self.precision = 0.0  # 精确率
        self.recall = 0.0  # 召回率
        self.f1_score = 0.0  # F1分数
        self.char_accuracy = 0.0  # 字符级准确率
        self.word_accuracy = 0.0  # 单词级准确率
```

## 数据流设计

### 图片处理流程

```
1. ImageLoader.load(image_path) → image_array
2. TextDetector.detect(image_array) → List[BoundingBox]
3. IF bboxes为空 THEN 返回空结果
4. TextRecognizer.recognize(image_array, bboxes) → List[RecognitionResult]
5. ResultFormatter.to_json(results) → json_output
```

**需求覆盖：** 1.1, 2.1, 3.1, 4.1, 4.2, 4.5, 7.1

### 视频处理流程

```
1. VideoProcessor.extract_frames(video_path) → Iterator[frame]
2. FOR EACH frame:
   a. TextDetector.detect(frame) → List[BoundingBox]
   b. IF bboxes非空 THEN
      TextRecognizer.recognize(frame, bboxes) → List[RecognitionResult]
   c. 收集结果
3. 返回所有帧的结果列表
```

**需求覆盖：** 1.2, 4.2, 7.4, 9.2

### 地铁站名识别流程

```
1. 加载地铁线路图图片
2. TextDetector.detect() → 检测所有站名区域
3. TextRecognizer.recognize() → 识别中英文站名
4. 后处理：
   - 解析中英文对应关系
   - 过滤低置信度结果（< 0.5）
   - 保持站名的空间位置关系
```

**需求覆盖：** 6.1, 6.2, 6.3, 6.4, 6.5

## 错误处理策略

### 错误类型和处理

```python
class OCRException(Exception):
    """OCR系统基础异常类"""
    pass

class FileLoadError(OCRException):
    """文件加载错误 - 需求 8.1, 8.2"""
    pass

class ModelLoadError(OCRException):
    """模型加载错误 - 需求 8.3"""
    pass

class InsufficientMemoryError(OCRException):
    """内存不足错误 - 需求 8.4"""
    pass

class UnsupportedFormatError(OCRException):
    """不支持的格式错误 - 需求 8.1"""
    pass
```

### 错误处理流程

1. 所有异常都继承自 `OCRException`
2. 每个异常包含描述性错误消息
3. 所有错误记录到日志文件（需求 8.5）
4. 向用户返回友好的错误消息

## 性能优化策略

### GPU加速（需求 9.3, 9.4）

```python
# 检测GPU可用性
if torch.cuda.is_available():
    device = 'gpu'
else:
    device = 'cpu'

# PaddleOCR初始化时指定设备
ocr = PaddleOCR(use_gpu=True, gpu_mem=500)
```

### 批量处理（需求 9.5）

```python
def batch_process(images: List[np.ndarray], batch_size: int = 8) -> List[OCRResult]:
    """
    批量处理图片以提高吞吐量
    
    参数:
        images: 图片列表
        batch_size: 批次大小
        
    返回:
        List[OCRResult]: 结果列表
    """
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = ocr.ocr(batch)
        results.extend(batch_results)
    return results
```

### 性能目标

- 单张1920x1080图片：< 5秒（需求 9.1）
- 视频处理：≥ 2 FPS（需求 9.2）
- 使用GPU时性能提升3-5倍

## 配置文件格式

```yaml
# config.yaml
detection:
  model_type: "DB"  # 可选: EAST, DB, CRAFT, PSENet
  model_path: "models/det_model"
  
recognition:
  model_type: "CRNN"  # 可选: CRNN, Tesseract, EasyOCR, PaddleOCR
  model_path: "models/rec_model"
  lang: "ch_en"  # 可选: ch, en, ch_en
  
system:
  use_gpu: true
  gpu_mem: 500  # MB
  confidence_threshold: 0.5
  output_format: "json"  # 可选: json, xml, csv
  log_level: "INFO"
  log_file: "ocr_system.log"
  
performance:
  batch_size: 8
  max_image_size: [1920, 1080]
```

**需求覆盖：** 10.1, 10.2, 10.3, 10.5

## 扩展性设计

### 模型插件接口（需求 10.4）

```python
class DetectorPlugin(ABC):
    """检测器插件接口"""
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        pass

class RecognizerPlugin(ABC):
    """识别器插件接口"""
    @abstractmethod
    def recognize(self, image: np.ndarray, bboxes: List[BoundingBox]) -> List[RecognitionResult]:
        pass

# 注册新模型
class CustomDetector(DetectorPlugin):
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        # 自定义检测逻辑
        pass

# 使用插件
detector = CustomDetector()
ocr_system = OCRSystem(detector=detector)
```

## 依赖项

### 核心依赖

- **PaddleOCR**: 主要OCR框架（推荐，中文支持好）
- **PaddlePaddle**: 深度学习框架
- **OpenCV**: 图像处理
- **NumPy**: 数组操作
- **Pillow**: 图像加载

### 可选依赖

- **EasyOCR**: 备选OCR框架
- **Tesseract**: 传统OCR引擎
- **PyTorch**: 如果使用PyTorch模型

### 安装命令

```bash
pip install paddlepaddle-gpu paddleocr opencv-python numpy pillow
```

## 测试策略

### 单元测试

- 测试每个组件的独立功能
- 使用mock对象隔离依赖
- 覆盖正常流程和异常情况

### 集成测试

- 测试端到端OCR流程
- 使用真实图片和视频
- 验证两阶段模型集成

### 精度测试

- 使用标注数据集评估精度
- 分别测试中文和英文识别
- 测试地铁站名识别场景

### 性能测试

- 测试处理时间是否满足要求
- 测试GPU加速效果
- 测试批量处理吞吐量

## 正确性属性

本系统主要涉及机器学习模型的推理和数据处理，不适合使用基于属性的测试（Property-Based Testing）。原因如下：

1. **非确定性输出**：OCR模型的输出依赖于训练数据和模型参数，对于相同输入可能产生不同的识别结果（尤其是置信度分数）
2. **无明确的数学不变量**：文字识别是一个复杂的感知任务，没有简单的数学属性可以验证
3. **依赖外部模型**：系统行为高度依赖PaddleOCR等外部模型的实现

因此，本系统将使用以下测试方法：

- **示例测试（Example-Based Testing）**：使用具体的测试图片和预期结果
- **精度评估**：使用标注数据集计算准确率、召回率等指标
- **集成测试**：验证各组件的正确集成和数据流
- **边界条件测试**：测试空图片、损坏文件、极端尺寸等边界情况

## 部署考虑

### 环境要求

- Python 3.7+
- CUDA 10.2+（如果使用GPU）
- 至少4GB内存（CPU模式）
- 至少2GB显存（GPU模式）

### 模型文件

- 检测模型：~10MB
- 识别模型：~20MB
- 总计：~30MB

### 容器化

```dockerfile
FROM python:3.8-slim
RUN pip install paddlepaddle-gpu paddleocr opencv-python
COPY . /app
WORKDIR /app
CMD ["python", "main.py"]
```

## 需求追溯矩阵

| 组件 | 覆盖的需求 |
|------|-----------|
| OCRSystem | 1.1, 1.2, 1.3, 4.1, 5.1-5.5 |
| ImageLoader | 1.1, 1.4, 8.1, 8.2 |
| VideoProcessor | 1.2, 1.5, 7.4, 8.1, 8.2 |
| StreamProcessor | 1.3 |
| TextDetector | 2.1, 2.2, 2.4, 2.5, 8.3, 9.3, 9.4 |
| TextRecognizer | 3.1-3.6, 4.2, 8.3, 9.3, 9.4 |
| ResultFormatter | 7.1, 7.2, 7.5, 10.5 |
| Config | 10.1, 10.2, 10.3, 10.5 |
| AccuracyEvaluator | 5.1-5.5 |
| 错误处理 | 8.1-8.5 |
| 性能优化 | 9.1-9.5 |
| 插件接口 | 10.4 |

## 总结

本设计采用模块化架构，将OCR系统分解为清晰的组件，每个组件负责特定的功能。使用PaddleOCR作为核心框架，提供了良好的中文支持和性能。系统支持多种输入源、可配置的模型选择、完善的错误处理和性能优化策略，满足所有功能和非功能需求。
