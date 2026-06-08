# OCR系统迁移至RK3576开发板需求文档

## 1. 项目概述

### 1.1 项目背景
当前OCR文字识别系统基于PaddleOCR框架开发，运行在GPU环境下，主要用于实时识别地铁站点等场景。为了实现边缘部署和降低成本，需要将系统迁移至瑞芯微RK3576开发板，利用其NPU（神经网络处理单元）进行推理加速。

### 1.2 当前系统概况

**技术栈：**
- 框架：PaddleOCR 2.10.0+
- 推理引擎：PaddlePaddle GPU 2.6.2+
- 编程语言：Python 3.10+
- 图像处理：OpenCV 4.8.0+

**核心功能：**
- 文本检测和识别（支持中英文）
- ROI区域检测（基于颜色检测）
- 多种输入模式：图片、视频、批量、流媒体（RTSP/RTMP）
- 站点名称匹配（模糊匹配）
- 结果可视化和持久化

**性能指标：**
- GPU推理速度：~30-50 FPS（视频流模式）
- 识别准确率：>90%（中文站点名）
- 置信度阈值：0.5

### 1.3 目标平台分析

**RK3576硬件规格：**
- CPU：8核ARM（4×Cortex-A72 + 4×Cortex-A53）
- NPU：6 TOPS算力，支持INT8/INT16量化
- 内存：4GB/8GB LPDDR4
- 支持框架：RKNN Toolkit2

**优势：**
- 低功耗（<10W）
- NPU专用加速，推理效率高
- 成本低，适合边缘部署
- 支持多种视频解码格式

**限制：**
- 内存和计算资源相比GPU受限
- 需要模型量化（INT8）
- RKNN框架生态相比PaddlePaddle较小

## 2. 技术挑战与解决方案

### 2.1 核心技术挑战

| 挑战 | 影响程度 | 优先级 |
|------|---------|--------|
| PaddleOCR模型转换为RKNN格式 | 高 | P0 |
| 模型量化导致的精度损失 | 高 | P0 |
| NPU推理接口适配 | 高 | P0 |
| Python依赖库兼容性（ARM架构） | 中 | P1 |
| 内存优化（4GB限制） | 中 | P1 |
| 视频流解码性能 | 中 | P2 |

### 2.2 技术方案

#### 2.2.1 模型转换方案

**方案A：PaddleOCR → ONNX → RKNN（推荐）**

```
PaddlePaddle模型 (.pdmodel/.pdiparams)
    ↓ Paddle2ONNX
ONNX模型 (.onnx)
    ↓ RKNN-Toolkit2
RKNN模型 (.rknn)
    ↓ 部署
RK3576 NPU推理
```

**优点：**
- ONNX是中间格式，工具链成熟
- RKNN-Toolkit2对ONNX支持较好
- 可在PC上完成转换和验证

**步骤：**
1. 导出PaddleOCR检测模型和识别模型为ONNX
2. 使用RKNN-Toolkit2转换为RKNN模型
3. 量化配置：INT8量化，准备校准数据集
4. 精度对比验证

**方案B：直接使用RKNN原生模型**

考虑使用瑞芯微官方提供的OCR模型（如果有），或者基于YOLO+CRNN自建检测识别模型。

**优点：** 无转换损失，原生支持好
**缺点：** 需重新训练模型，开发周期长

**建议：** 优先采用方案A，如果精度或性能不达标，再考虑方案B

#### 2.2.2 推理框架适配

创建统一的推理抽象层，支持多种后端：

```python
# 推理引擎抽象接口
class InferenceEngine(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        pass

# PaddleOCR引擎（当前实现）
class PaddleOCREngine(InferenceEngine):
    def __init__(self, use_gpu=True):
        self.ocr = PaddleOCR(use_gpu=use_gpu)
    ...

# RKNN引擎（新增）
class RKNNEngine(InferenceEngine):
    def __init__(self):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
    
    def load_model(self, model_path: str):
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.rknn.inference(inputs=[input_data])
```

**配置驱动选择引擎：**
```yaml
system:
  inference_engine: "rknn"  # 可选: paddleocr, rknn
  device: "npu"  # 可选: gpu, cpu, npu
```

#### 2.2.3 性能优化策略

**1. 模型优化**
- INT8量化（精度损失<2%）
- 输入分辨率优化（640×640 → 512×512）
- 检测模型和识别模型分离部署

**2. 内存优化**
- 视频帧缓冲池复用
- 大图分块处理
- ROI裁剪优先处理（减少无效区域）

**3. 多线程并行**
```
线程1: 视频解码
   ↓
线程2: ROI检测（CPU）
   ↓
线程3: NPU推理（检测模型）
   ↓
线程4: NPU推理（识别模型）
   ↓
线程5: 后处理（站点匹配）
```

**4. 跳帧策略**
- 动态调整frame_interval（根据CPU负载）
- 变化检测：只有画面变化时才推理

## 3. 详细任务分解

### 3.1 阶段一：环境搭建与评估（1周）

**任务1.1：RK3576开发环境搭建**
- [ ] 安装Ubuntu 20.04系统（RK3576官方镜像）
- [ ] 配置RKNN Toolkit2开发环境
- [ ] 安装Python 3.8+及依赖库（OpenCV、NumPy等）
- [ ] 验证NPU可用性（运行官方demo）

**任务1.2：模型转换工具链搭建**
- [ ] 在PC端安装RKNN-Toolkit2（x86_64）
- [ ] 安装Paddle2ONNX工具
- [ ] 准备模型转换脚本

**任务1.3：基准测试**
- [ ] 测试PaddleOCR模型在CPU模式下的性能
- [ ] 测试RK3576 NPU的理论算力
- [ ] 建立性能baseline

**交付物：**
- RK3576开发环境配置文档
- 基准测试报告（性能数据）

### 3.2 阶段二：模型转换与验证（2周）

**任务2.1：导出PaddleOCR模型**
```python
# 检测模型导出
from paddleocr import PaddleOCR
ocr = PaddleOCR()
# 导出检测模型为ONNX
paddle2onnx --model_dir det_model \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file det_model.onnx \
            --opset_version 11

# 识别模型导出
paddle2onnx --model_dir rec_model \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file rec_model.onnx \
            --opset_version 11
```

**任务2.2：ONNX转RKNN**
```python
from rknn.api import RKNN

rknn = RKNN()

# 配置
rknn.config(
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[127.5, 127.5, 127.5]],
    target_platform='rk3576',
    quantized_dtype='asymmetric_quantized-8'  # INT8量化
)

# 加载ONNX
rknn.load_onnx(model='det_model.onnx')

# 构建RKNN模型
rknn.build(do_quantization=True, dataset='calibration_dataset.txt')

# 导出
rknn.export_rknn('det_model.rknn')
```

**任务2.3：精度验证**
- [ ] 准备测试数据集（100张地铁站点图片）
- [ ] 对比PaddleOCR vs RKNN的识别结果
- [ ] 计算精度损失（字符错误率CER、准确率）
- [ ] 精度不达标则调整量化参数

**验收标准：**
- 检测模型召回率 ≥ 95%
- 识别模型准确率 ≥ 88%（允许2-5%精度损失）
- 端到端识别准确率 ≥ 85%

**交付物：**
- 转换后的RKNN模型文件（det_model.rknn, rec_model.rknn）
- 精度对比报告
- 模型转换脚本

### 3.3 阶段三：推理引擎适配（2周）

**任务3.1：创建RKNN推理引擎**

新增文件：`src/inference/rknn_engine.py`
```python
"""RKNN推理引擎"""
from rknnlite.api import RKNNLite
import numpy as np
from typing import List
from ..models import BoundingBox, RecognitionResult

class RKNNDetector:
    """基于RKNN的文本检测器"""
    def __init__(self, model_path: str):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        assert ret == 0, "加载RKNN模型失败"
        ret = self.rknn.init_runtime()
        assert ret == 0, "初始化RKNN运行时失败"
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        # 预处理
        input_data = self.preprocess(image)
        # NPU推理
        outputs = self.rknn.inference(inputs=[input_data])
        # 后处理（DB算法）
        bboxes = self.postprocess(outputs[0])
        return bboxes
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # 缩放、归一化
        resized = cv2.resize(image, (640, 640))
        normalized = (resized - 127.5) / 127.5
        return normalized.astype(np.float32)
    
    def postprocess(self, output: np.ndarray) -> List[BoundingBox]:
        # DB后处理：二值化 → 轮廓提取 → bbox
        ...

class RKNNRecognizer:
    """基于RKNN的文本识别器"""
    def __init__(self, model_path: str, dict_path: str):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime()
        self.char_dict = self.load_dict(dict_path)
    
    def recognize(self, image: np.ndarray, bboxes: List[BoundingBox]) -> List[RecognitionResult]:
        results = []
        for bbox in bboxes:
            # 裁剪ROI
            roi = self.crop_region(image, bbox)
            # 预处理
            input_data = self.preprocess(roi)
            # 推理
            outputs = self.rknn.inference(inputs=[input_data])
            # 解码（CTC Decode）
            text, confidence = self.decode(outputs[0])
            results.append(RecognitionResult(text, confidence, bbox))
        return results
```

**任务3.2：修改OCRSystem集成RKNN引擎**

修改文件：`src/ocr_system.py`
```python
def __init__(self, config: Optional[Config] = None):
    self.config = config if config else Config()
    
    # 根据配置选择推理引擎
    if self.config.inference_engine == "rknn":
        from .inference.rknn_engine import RKNNDetector, RKNNRecognizer
        self.text_detector = RKNNDetector(
            model_path=self.config.rknn_det_model
        )
        self.text_recognizer = RKNNRecognizer(
            model_path=self.config.rknn_rec_model,
            dict_path=self.config.rknn_dict
        )
    elif self.config.inference_engine == "paddleocr":
        from .text_detector import TextDetector
        from .text_recognizer import TextRecognizer
        self.text_detector = TextDetector(use_gpu=self.config.use_gpu)
        self.text_recognizer = TextRecognizer(
            lang=self.config.rec_lang, 
            use_gpu=self.config.use_gpu
        )
    else:
        raise ValueError(f"不支持的推理引擎: {self.config.inference_engine}")
```

**任务3.3：配置文件扩展**

修改文件：`config/config.yaml`
```yaml
# 系统配置
system:
  inference_engine: "rknn"  # 推理引擎: paddleocr, rknn
  device: "npu"  # 设备: gpu, cpu, npu
  use_gpu: false  # RKNN模式下此项无效

# RKNN配置（仅当inference_engine=rknn时生效）
rknn:
  det_model: "models/det_model.rknn"
  rec_model: "models/rec_model.rknn"
  dict_file: "models/ppocr_keys_v1.txt"
  core_mask: 0  # NPU核心掩码（0=自动, 1=core0, 2=core1, 3=core0+1）
```

**交付物：**
- RKNN推理引擎代码
- 修改后的OCRSystem
- 单元测试（测试RKNN引擎功能）

### 3.4 阶段四：性能优化与测试（2周）

**任务4.1：推理性能优化**
- [ ] 多线程并行（视频解码、NPU推理、后处理）
- [ ] 内存池复用（减少malloc开销）
- [ ] 批处理推理（如果内存允许）

**任务4.2：系统集成测试**
测试场景：
1. 图片模式：批量测试100张图片，统计准确率和速度
2. 视频模式：测试30分钟视频，统计FPS和准确率
3. 流媒体模式：测试RTMP流，连续运行4小时，检查稳定性
4. 压力测试：1080P@30FPS持续运行，监控内存和CPU占用

**任务4.3：边界条件测试**
- [ ] 低光照场景
- [ ] 模糊图像
- [ ] 倾斜文字
- [ ] 多行文字
- [ ] 中英文混合

**验收标准：**
- 推理速度：≥20 FPS（720P视频流）
- 内存占用：<3GB
- CPU占用：<60%
- 准确率：≥85%（端到端）
- 稳定性：连续运行24小时无崩溃

**交付物：**
- 性能优化报告
- 集成测试报告
- 压力测试报告

### 3.5 阶段五：部署与文档（1周）

**任务5.1：部署脚本**
```bash
#!/bin/bash
# deploy.sh - RK3576一键部署脚本

# 1. 安装依赖
apt-get update
apt-get install -y python3-pip python3-opencv librknn-api

# 2. 安装Python依赖
pip3 install -r requirements_rk3576.txt

# 3. 复制模型文件
mkdir -p models
cp det_model.rknn models/
cp rec_model.rknn models/

# 4. 配置权限
chmod +x main.py

# 5. 启动服务
python3 main.py --config config/config_rk3576.yaml
```

**任务5.2：编写文档**
- [ ] RK3576部署手册（环境搭建、依赖安装、配置说明）
- [ ] 模型转换指南（详细步骤和常见问题）
- [ ] API文档（RKNN引擎接口说明）
- [ ] 故障排查指南（常见错误和解决方案）

**任务5.3：示例代码**
```python
# examples/rk3576_demo.py
"""RK3576平台OCR示例"""
from src.config import Config
from src.ocr_system import OCRSystem

# 加载RK3576配置
config = Config("config/config_rk3576.yaml")

# 初始化OCR系统（自动使用RKNN引擎）
ocr = OCRSystem(config)

# 处理图片
result = ocr.process_image("test.jpg")
for r in result.results:
    print(f"文字: {r.text}, 置信度: {r.confidence:.2f}")

# 处理视频流
for frame_result in ocr.process_stream("rtmp://server/live/stream"):
    # 实时处理逻辑
    ...
```

**交付物：**
- 部署脚本
- 完整技术文档（30+页）
- 示例代码

## 4. 依赖与资源

### 4.1 硬件资源
- RK3576开发板 × 1（8GB内存版本推荐）
- PC（用于模型转换）：Intel i5+, 16GB RAM, Ubuntu 20.04
- 测试摄像头/RTMP流源

### 4.2 软件依赖

**RK3576开发板（ARM64）：**
```txt
# requirements_rk3576.txt
python>=3.8
opencv-python>=4.5.0
numpy>=1.21.0
PyYAML>=6.0
Pillow>=9.0.0
# RKNN运行时（通过apt安装，不在pip中）
# librknnrt.so
```

**PC端（模型转换）：**
```txt
# requirements_conversion.txt
rknn-toolkit2>=1.6.0
paddle2onnx>=1.0.0
paddlepaddle>=2.5.0
onnx>=1.12.0
onnxruntime>=1.13.0
```

### 4.3 人力资源
- 算法工程师 × 1（负责模型转换和优化）
- 嵌入式开发工程师 × 1（负责系统集成和性能优化）
- 测试工程师 × 1（负责功能测试和性能测试）

### 4.4 时间计划
总周期：**8周**
- 阶段一：1周
- 阶段二：2周
- 阶段三：2周
- 阶段四：2周
- 阶段五：1周

## 5. 风险评估与应对

### 5.1 技术风险

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 模型转换失败 | 中 | 高 | 提前验证转换工具链；准备替代方案（使用其他OCR模型） |
| 量化精度损失过大 | 高 | 高 | 扩充校准数据集；尝试混合量化（部分FP16） |
| NPU算力不足 | 中 | 高 | 模型裁剪；降低输入分辨率；使用轻量级模型 |
| 内存不足（4GB） | 中 | 中 | 内存优化；降低batch_size；使用swap |
| RKNN运行时bug | 低 | 高 | 联系瑞芯微技术支持；使用CPU fallback |

### 5.2 进度风险

**风险：** 模型转换和精度调优耗时超预期
**应对：** 
- 预留1周缓冲时间
- 精度要求可适当放宽（85% → 80%）
- 必要时采用增量迁移（先迁移检测模型，识别模型保持CPU推理）

### 5.3 依赖风险

**风险：** RKNN-Toolkit2版本更新导致兼容性问题
**应对：** 
- 锁定RKNN-Toolkit2版本号
- 建立Docker镜像固化环境
- 保留转换好的RKNN模型备份

## 6. 验收标准

### 6.1 功能验收
- [ ] 支持图片、视频、流媒体三种输入模式
- [ ] ROI检测功能正常
- [ ] 站点匹配功能正常
- [ ] 结果可视化和保存功能正常

### 6.2 性能验收
- [ ] 推理速度：720P视频流 ≥20 FPS
- [ ] 内存占用：<3GB（4小时运行）
- [ ] CPU占用：<60%（平均值）
- [ ] 功耗：<10W

### 6.3 精度验收
- [ ] 检测召回率：≥95%
- [ ] 识别准确率：≥88%（单字准确率）
- [ ] 端到端准确率：≥85%（完整站点名）
- [ ] 对比GPU版本精度损失：<5%

### 6.4 稳定性验收
- [ ] 连续运行24小时无崩溃
- [ ] 内存无泄漏（内存占用稳定）
- [ ] 异常处理完善（网络断开、模型加载失败等）

### 6.5 文档验收
- [ ] 部署手册完整可执行
- [ ] 模型转换指南清晰
- [ ] 代码注释完整
- [ ] API文档齐全

## 7. 后续优化方向

### 7.1 短期优化（3个月内）
1. **模型蒸馏**：训练更小的学生模型，进一步提升速度
2. **多尺度推理**：根据文字大小动态调整输入分辨率
3. **缓存机制**：对连续帧中不变的区域跳过推理

### 7.2 长期优化（6个月+）
1. **端到端模型**：检测+识别融合为单模型，减少推理次数
2. **在线学习**：根据实际场景持续优化模型
3. **多模态融合**：结合时序信息提升识别准确率
4. **硬件加速**：利用RK3576的视频硬解码器

## 8. 参考资料

### 8.1 官方文档
- [RKNN Toolkit2用户指南](https://github.com/rockchip-linux/rknn-toolkit2)
- [PaddleOCR文档](https://github.com/PaddlePaddle/PaddleOCR)
- [Paddle2ONNX转换工具](https://github.com/PaddlePaddle/Paddle2ONNX)

### 8.2 技术博客
- 《PaddleOCR模型转RKNN实战》
- 《RK3576 NPU性能优化指南》
- 《INT8量化精度损失分析》

### 8.3 社区资源
- 瑞芯微开发者论坛
- PaddleOCR GitHub Issues
- RKNN-Toolkit2 示例代码

---

**文档版本：** v1.0  
**创建日期：** 2026-06-03  
**创建人：** AI Assistant  
**审核状态：** 待审核
