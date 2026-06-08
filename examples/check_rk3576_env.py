"""
RK3576开发板环境检查脚本
检查RKNN运行时库、Python依赖等
"""
import sys
import os

def check_python_version():
    """检查Python版本"""
    print("\n" + "="*60)
    print("Python版本检查")
    print("="*60)
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✓ Python版本符合要求 (>=3.8)")
        return True
    else:
        print("✗ Python版本过低，需要 >=3.8")
        return False

def check_package(package_name, import_name=None):
    """检查Python包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

def check_rknnlite():
    """检查RKNN运行时库"""
    print("\n" + "="*60)
    print("RKNN运行时库检查")
    print("="*60)

    try:
        from rknnlite.api import RKNNLite
        print(f"✓ rknnlite 已安装")

        # 尝试创建实例
        try:
            rknn = RKNNLite()
            print(f"✓ RKNNLite 实例创建成功")
            del rknn
            return True
        except Exception as e:
            print(f"✗ RKNNLite 实例创建失败: {e}")
            return False

    except ImportError:
        print(f"✗ rknnlite 未安装")
        print("\n安装方法:")
        print("  sudo apt-get update")
        print("  sudo apt-get install python3-rknnlite2")
        return False

def check_dependencies():
    """检查其他依赖"""
    print("\n" + "="*60)
    print("依赖库检查")
    print("="*60)

    packages = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'Pillow': 'PIL',
    }

    results = {}
    for package_name, import_name in packages.items():
        results[package_name] = check_package(package_name, import_name)

    return all(results.values())

def check_npu_device():
    """检查NPU设备"""
    print("\n" + "="*60)
    print("NPU设备检查")
    print("="*60)

    # 检查设备文件
    npu_devices = ['/dev/rknpu', '/dev/dri/renderD128']

    found = False
    for device in npu_devices:
        if os.path.exists(device):
            print(f"✓ NPU设备存在: {device}")
            found = True
        else:
            print(f"  {device} 不存在")

    if not found:
        print("⚠ 未找到NPU设备文件")
        print("  这可能是正常的，某些系统配置不同")

    return True  # 不强制要求

def check_model_files():
    """检查模型文件"""
    print("\n" + "="*60)
    print("模型文件检查")
    print("="*60)

    # 支持多个可能的路径
    possible_dirs = [
        "rknn",
        "models/rknn",
        "/home/rk/ai_ocr/rknn"
    ]

    model_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            model_dir = dir_path
            print(f"✓ 找到模型目录: {model_dir}")
            break

    if not model_dir:
        print(f"✗ 模型目录不存在")
        print(f"  查找路径: {possible_dirs}")
        print(f"  请创建目录: mkdir -p rknn")
        return False

    models = [
        "ch_PP-OCRv4_det_rk3576.rknn",
        "ch_PP-OCRv4_rec_rk3576.rknn",
        "ch_ppocr_mobile_v2.0_cls_rk3576.rknn"
    ]

    all_exist = True
    for model in models:
        model_path = os.path.join(model_dir, model)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ {model} ({size:.2f} MB)")
        else:
            print(f"✗ {model} 不存在")
            all_exist = False

    if not all_exist:
        print(f"\n请将RKNN模型文件复制到 {model_dir}/ 目录")

    return all_exist

def check_dict_file():
    """检查字符字典文件"""
    print("\n" + "="*60)
    print("字符字典检查")
    print("="*60)

    # 支持多个可能的路径
    possible_paths = [
        "rknn/ppocr_keys_v1.txt",
        "models/ppocr_keys_v1.txt",
        "/home/rk/ai_ocr/rknn/ppocr_keys_v1.txt"
    ]

    dict_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dict_path = path
            break

    if dict_path:
        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✓ {dict_path} ({len(lines)} 个字符)")
        return True
    else:
        print(f"✗ 字典文件不存在")
        print(f"  查找路径: {possible_paths}")
        print("\n字典文件可以从PaddleOCR获取:")
        print("  ~/.paddleocr/whl/rec/ppocr_keys_v1.txt")
        return False

def main():
    """主函数"""
    print("="*60)
    print("RK3576开发板环境检查")
    print("="*60)

    results = {
        'Python版本': check_python_version(),
        'RKNN运行时': check_rknnlite(),
        '依赖库': check_dependencies(),
        'NPU设备': check_npu_device(),
        '模型文件': check_model_files(),
        '字符字典': check_dict_file(),
    }

    # 总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)

    for item, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {item}")

    # 关键项
    critical_items = ['Python版本', 'RKNN运行时', '依赖库', '模型文件', '字符字典']
    critical_passed = all(results[item] for item in critical_items)

    print("\n" + "="*60)
    if critical_passed:
        print("✓ 环境检查通过，可以开始推理测试！")
        print("\n下一步:")
        print("  python examples/test_rknn_inference.py test_image.jpg")
        return 0
    else:
        print("✗ 环境检查失败，请修复上述问题")
        print("\n常见问题:")
        print("  1. 安装rknnlite: sudo apt-get install python3-rknnlite2")
        print("  2. 安装依赖: pip install numpy opencv-python Pillow")
        print("  3. 复制模型: scp models/rknn/*.rknn rk@rk3576:~/ai_ocr/models/rknn/")
        print("  4. 复制字典: 从PaddleOCR目录获取")
        return 1

if __name__ == "__main__":
    sys.exit(main())
