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
]

results = []
for pkg in packages:
    results.append(check_package(pkg))

# 单独检查RKNN（仅Linux）
rknn_available = False
if sys.platform == "linux":
    try:
        from rknn.api import RKNN
        print(f"✓ rknn.api 安装成功")
        rknn_available = True
    except ImportError:
        print(f"✗ rknn.api 未安装（Windows下正常，需要WSL或Docker）")

print("=" * 50)
if all(results):
    print("✓ PC端转换环境搭建完成")
    if sys.platform != "linux":
        print("⚠ 注意: RKNN-Toolkit2需要Linux环境，请使用WSL2或Docker")
else:
    print("✗ 请安装缺失的包")
print("=" * 50)

# 测试版本信息
print("\n版本信息:")
try:
    import paddle2onnx
    print(f"Paddle2ONNX: {paddle2onnx.__version__}")
except:
    pass

try:
    import onnx
    print(f"ONNX: {onnx.__version__}")
except:
    pass

try:
    import paddle
    print(f"PaddlePaddle: {paddle.__version__}")
except:
    pass

try:
    import paddleocr
    print(f"PaddleOCR: {paddleocr.__version__}")
except:
    pass

# 测试RKNN-Toolkit2（仅Linux）
if rknn_available:
    print("\n测试RKNN-Toolkit2...")
    try:
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        print(f"✓ RKNN实例创建成功")
        del rknn
    except Exception as e:
        print(f"✗ RKNN-Toolkit2 测试失败: {e}")
