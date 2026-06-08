# 第一步：PC端环境搭建指南

## 前置条件
- 操作系统：Ubuntu 20.04 / 22.04（推荐）或 Windows 10/11
- Python 3.8-3.10
- 至少16GB RAM
- 50GB可用磁盘空间

## 任务清单

### 1. 创建Python虚拟环境

```bash
# 在项目根目录
cd D:/AI_code/ai_ocr

# 创建转换工具专用虚拟环境
python -m venv venv_conversion

# 激活虚拟环境
# Windows:
venv_conversion\Scripts\activate
# Linux/Mac:
source venv_conversion/bin/activate
```

### 2. 安装模型转换工具

```bash
# 安装Paddle2ONNX
pip install paddle2onnx==1.0.9

# 安装ONNX相关工具
pip install onnx==1.14.0 onnxruntime==1.15.0

# 如果需要可视化ONNX模型
pip install netron

# 安装PaddlePaddle（用于导出模型）
pip install paddlepaddle==2.6.2
pip install paddleocr==2.10.0
```

### 3. 下载RKNN-Toolkit2（重要）

**选项A：Linux/Mac**
```bash
# 1. 从GitHub下载RKNN-Toolkit2
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
cd rknn-toolkit2/rknn-toolkit2/packages

# 2. 安装对应Python版本的whl包
# Python 3.8:
pip install rknn_toolkit2-1.6.0+81f21f4d-cp38-cp38-linux_x86_64.whl

# Python 3.9:
pip install rknn_toolkit2-1.6.0+81f21f4d-cp39-cp39-linux_x86_64.whl

# Python 3.10:
pip install rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl
```

**选项B：Windows（需要WSL或Docker）**
```powershell
# RKNN-Toolkit2不支持Windows原生环境
# 方案1：使用WSL2
wsl --install Ubuntu-22.04

# 方案2：使用Docker
docker pull ubuntu:22.04
docker run -it -v D:/AI_code/ai_ocr:/workspace ubuntu:22.04 /bin/bash
# 然后在容器内按照Linux步骤操作
```

### 4. 验证安装

创建测试脚本 `tools/test_conversion_env.py`：
```python
"""验证模型转换环境是否正确安装"""
import sys

def check_package(package_name):
    try:
        __import__(package_name)
        print(f"✓ {package_name} 安装成功")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

print("=" * 50)
print("检查模型转换环境")
print("=" * 50)

packages = [
    "paddle2onnx",
    "onnx",
    "onnxruntime",
    "paddle",
    "paddleocr",
    "rknn.api"  # 如果是Linux
]

results = []
for pkg in packages:
    results.append(check_package(pkg))

print("=" * 50)
if all(results[:5]):  # 前5个是必需的
    print("✓ PC端转换环境搭建完成")
else:
    print("✗ 请安装缺失的包")
print("=" * 50)

# 测试ONNX转换
print("\n测试Paddle2ONNX...")
import paddle2onnx
print(f"Paddle2ONNX版本: {paddle2onnx.__version__}")

# 测试RKNN-Toolkit2（仅Linux）
if sys.platform == "linux":
    try:
        from rknn.api import RKNN
        print(f"✓ RKNN-Toolkit2 加载成功")
        rknn = RKNN(verbose=False)
        print(f"✓ RKNN实例创建成功")
    except Exception as e:
        print(f"✗ RKNN-Toolkit2 测试失败: {e}")
```

运行验证：
```bash
python tools/test_conversion_env.py
```

### 5. 导出当前PaddleOCR模型

创建模型导出脚本 `tools/export_paddleocr_models.py`：
```python
"""导出PaddleOCR预训练模型"""
import os
from paddleocr import PaddleOCR

def export_models():
    """导出检测模型和识别模型"""
    
    # 创建输出目录
    os.makedirs("models/paddle_models", exist_ok=True)
    
    print("初始化PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        use_gpu=False,  # 导出时不需要GPU
        show_log=True
    )
    
    print("=" * 50)
    print("PaddleOCR模型已下载到:")
    print("~/.paddleocr/whl/")
    print("=" * 50)
    print("\n检测模型路径示例:")
    print("~/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/")
    print("\n识别模型路径示例:")
    print("~/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/")
    print("=" * 50)
    
    # 测试OCR
    print("\n测试OCR推理...")
    result = ocr.ocr("test_image.jpg", cls=True)
    if result:
        print("✓ OCR推理测试成功")
    else:
        print("✗ OCR推理测试失败")

if __name__ == "__main__":
    export_models()
```

### 6. 准备校准数据集

模型量化需要校准数据集（100-500张代表性图片）：

```bash
# 创建校准数据集目录
mkdir -p dataset/calibration

# 从测试集中选取代表性图片（覆盖不同场景）
# - 不同光照条件
# - 不同文字大小
# - 不同背景复杂度
# - 中英文混合

# 生成校准数据集列表文件
ls dataset/calibration/*.jpg > dataset/calibration_list.txt
```

## 完成标志

- [ ] Python虚拟环境创建成功
- [ ] Paddle2ONNX安装成功
- [ ] ONNX工具链安装成功
- [ ] RKNN-Toolkit2安装成功（Linux）
- [ ] 验证脚本运行通过
- [ ] PaddleOCR模型已下载
- [ ] 校准数据集准备完成（100+张图片）

## 预计耗时
2-4小时（取决于网络速度和系统环境）

## 常见问题

### Q1: RKNN-Toolkit2安装失败
**A:** 
- 确认系统是Linux x86_64
- 检查Python版本（3.8-3.10）
- 从官方GitHub下载最新版本

### Q2: Paddle2ONNX转换报错
**A:**
- 确认PaddlePaddle和Paddle2ONNX版本匹配
- 查看转换日志，定位不支持的算子

### Q3: Windows下无法安装RKNN-Toolkit2
**A:**
- 必须使用WSL2或Docker
- 推荐使用Ubuntu 22.04镜像

## 下一步
完成PC端环境搭建后，进入**模型转换阶段**（step2_model_conversion.md）
