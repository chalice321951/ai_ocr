"""将ONNX模型转换为RKNN格式（RK3576专用）"""
import os
import sys
import numpy as np
from pathlib import Path

def prepare_calibration_dataset(image_list_file, output_file, target_size=(640, 640)):
    """
    准备校准数据集

    Args:
        image_list_file: 图片路径列表文件
        output_file: 输出的校准数据集文件
        target_size: 目标尺寸 (width, height)
    """
    import cv2

    print(f"\n准备校准数据集...")
    print(f"图片列表: {image_list_file}")

    if not os.path.exists(image_list_file):
        print(f"✗ 校准数据集列表文件不存在: {image_list_file}")
        print(f"  请创建该文件，每行一个图片路径")
        return False

    # 读取图片路径
    with open(image_list_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if len(image_paths) == 0:
        print(f"✗ 校准数据集为空")
        return False

    print(f"  找到 {len(image_paths)} 张校准图片")

    # 写入校准数据集文件
    with open(output_file, 'w') as f:
        for img_path in image_paths:
            if os.path.exists(img_path):
                f.write(f"{img_path}\n")

    print(f"✓ 校准数据集准备完成: {output_file}")
    return True

def convert_onnx_to_rknn(onnx_path, output_path, model_name,
                         do_quantization=True,
                         dataset_path=None,
                         target_platform='rk3576'):
    """
    转换ONNX模型为RKNN格式

    Args:
        onnx_path: ONNX模型路径
        output_path: 输出RKNN模型路径
        model_name: 模型名称（用于日志）
        do_quantization: 是否量化
        dataset_path: 校准数据集路径（量化时需要）
        target_platform: 目标平台
    """
    print(f"\n{'='*60}")
    print(f"转换 {model_name}")
    print(f"{'='*60}")
    print(f"输入: {onnx_path}")
    print(f"输出: {output_path}")
    print(f"量化: {'是' if do_quantization else '否'}")
    print(f"平台: {target_platform}")

    if not os.path.exists(onnx_path):
        print(f"✗ ONNX模型不存在: {onnx_path}")
        return False

    try:
        from rknn.api import RKNN

        # 创建RKNN对象
        rknn = RKNN(verbose=True)

        # 配置模型
        print(f"\n配置RKNN模型...")
        ret = rknn.config(
            mean_values=[[127.5, 127.5, 127.5]],  # 归一化均值
            std_values=[[127.5, 127.5, 127.5]],   # 归一化标准差
            target_platform=target_platform,
            quantized_dtype='asymmetric_quantized-8' if do_quantization else 'float16',
            optimization_level=3
        )

        if ret != 0:
            print(f"✗ RKNN配置失败")
            return False

        print(f"✓ RKNN配置成功")

        # 加载ONNX模型（需要指定固定输入尺寸）
        print(f"\n加载ONNX模型...")

        # 根据不同模型设置输入尺寸
        if 'det' in model_name.lower():
            # 检测模型：640x640
            input_size_list = [[1, 3, 640, 640]]
            print(f"  检测模型输入尺寸: [1, 3, 640, 640]")
        elif 'rec' in model_name.lower():
            # 识别模型：48x320
            input_size_list = [[1, 3, 48, 320]]
            print(f"  识别模型输入尺寸: [1, 3, 48, 320]")
        elif 'cls' in model_name.lower():
            # 方向分类器：48x192
            input_size_list = [[1, 3, 48, 192]]
            print(f"  方向分类器输入尺寸: [1, 3, 48, 192]")
        else:
            # 默认尺寸
            input_size_list = [[1, 3, 640, 640]]
            print(f"  使用默认输入尺寸: [1, 3, 640, 640]")

        ret = rknn.load_onnx(
            model=onnx_path,
            input_size_list=input_size_list
        )

        if ret != 0:
            print(f"✗ 加载ONNX模型失败")
            return False

        print(f"✓ ONNX模型加载成功")

        # 构建RKNN模型
        print(f"\n构建RKNN模型...")
        ret = rknn.build(
            do_quantization=do_quantization,
            dataset=dataset_path if do_quantization else None
        )

        if ret != 0:
            print(f"✗ 构建RKNN模型失败")
            rknn.release()
            return False

        print(f"✓ RKNN模型构建成功")

        # 导出RKNN模型
        print(f"\n导出RKNN模型...")
        ret = rknn.export_rknn(output_path)

        if ret != 0:
            print(f"✗ 导出RKNN模型失败")
            rknn.release()
            return False

        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✓ 导出成功！")
        print(f"  文件大小: {file_size:.2f} MB")

        # 释放资源
        rknn.release()

        return True

    except Exception as e:
        print(f"✗ 转换出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*60)
    print("ONNX模型转RKNN工具 (RK3576)")
    print("="*60)

    # 检查RKNN-Toolkit2是否安装
    try:
        from rknn.api import RKNN
        print("✓ RKNN-Toolkit2 已安装")
    except ImportError:
        print("✗ RKNN-Toolkit2 未安装")
        print("\n请先安装RKNN-Toolkit2:")
        print("  git clone https://github.com/airockchip/rknn-toolkit2.git")
        print("  cd rknn-toolkit2/rknn-toolkit2/packages")
        print("  pip install rknn_toolkit2-*.whl")
        return 1

    # 自动查找校准数据集（支持多个可能的位置）
    possible_calibration_paths = [
        "ocr/calibration_list.txt",
        "/home/admin123/wjj/ocr/calibration_list.txt",
        "dataset/calibration_list.txt",
    ]

    calibration_list = None
    for path in possible_calibration_paths:
        if os.path.exists(path):
            calibration_list = path
            print(f"✓ 找到校准数据集: {path}")
            break

    if calibration_list:
        calibration_dataset = calibration_list  # 直接使用列表文件
    else:
        print(f"\n⚠ 警告: 未找到校准数据集")
        print(f"  查找路径: {possible_calibration_paths}")
        print(f"  将使用非量化模式（FP16），性能会较差")
        calibration_dataset = None

    # 自动查找ONNX模型目录
    possible_model_dirs = [
        "tool/models/onnx",
        "models/onnx",
        "../tool/models/onnx",
    ]

    onnx_dir = None
    for path in possible_model_dirs:
        if os.path.exists(path):
            onnx_dir = path
            print(f"✓ 找到ONNX模型目录: {path}")
            break

    if not onnx_dir:
        print(f"✗ 未找到ONNX模型目录")
        print(f"  查找路径: {possible_model_dirs}")
        return 1

    # 创建输出目录（与ONNX同级）
    rknn_dir = os.path.join(os.path.dirname(onnx_dir), "rknn")
    os.makedirs(rknn_dir, exist_ok=True)
    print(f"✓ RKNN输出目录: {rknn_dir}")

    # 定义要转换的模型（使用fixed版本）
    models = {
        "检测模型 (ch_PP-OCRv4_det)": {
            "onnx": os.path.join(onnx_dir, "ch_PP-OCRv4_det_fixed.onnx"),
            "rknn": os.path.join(rknn_dir, "ch_PP-OCRv4_det_rk3576.rknn")
        },
        "识别模型 (ch_PP-OCRv4_rec)": {
            "onnx": os.path.join(onnx_dir, "ch_PP-OCRv4_rec_fixed.onnx"),
            "rknn": os.path.join(rknn_dir, "ch_PP-OCRv4_rec_rk3576.rknn")
        },
        "方向分类器 (ch_ppocr_mobile_v2.0_cls)": {
            "onnx": os.path.join(onnx_dir, "ch_ppocr_mobile_v2.0_cls_fixed.onnx"),
            "rknn": os.path.join(rknn_dir, "ch_ppocr_mobile_v2.0_cls_rk3576.rknn")
        }
    }

    # 转换所有模型
    results = {}
    for model_name, paths in models.items():
        onnx_path = paths["onnx"]
        rknn_path = paths["rknn"]

        # 转换
        do_quantization = (calibration_dataset is not None)
        success = convert_onnx_to_rknn(
            onnx_path=onnx_path,
            output_path=rknn_path,
            model_name=model_name,
            do_quantization=do_quantization,
            dataset_path=calibration_dataset,
            target_platform='rk3576'
        )

        results[model_name] = success

    # 打印总结
    print("\n" + "="*60)
    print("转换总结")
    print("="*60)
    for model_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{status} - {model_name}")

    all_success = all(results.values())
    if all_success:
        print("\n✓ 所有模型转换成功！")
        print("\nRKNN模型已保存到: models/rknn/")
        print("\n下一步: 将RKNN模型部署到RK3576开发板")
        print("  1. 将 models/rknn/*.rknn 复制到开发板")
        print("  2. 安装 rknnlite 运行时库")
        print("  3. 运行推理测试")
    else:
        print("\n✗ 部分模型转换失败，请检查错误信息")
        return 1

    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
