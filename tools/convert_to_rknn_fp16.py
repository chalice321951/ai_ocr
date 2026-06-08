"""
转换RKNN模型（不量化，FP16模式）
用于测试是否是量化导致的问题
"""
import os
import sys
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_onnx_to_rknn_fp16(onnx_path, output_path, model_name, target_platform='rk3576'):
    """
    转换ONNX模型为RKNN格式（FP16，不量化）
    """
    print(f"\n{'='*60}")
    print(f"转换 {model_name}")
    print(f"{'='*60}")
    print(f"输入: {onnx_path}")
    print(f"输出: {output_path}")
    print(f"量化: 否 (FP16)")
    print(f"平台: {target_platform}")

    if not os.path.exists(onnx_path):
        print(f"✗ ONNX模型不存在: {onnx_path}")
        return False

    try:
        from rknn.api import RKNN

        # 创建RKNN对象
        rknn = RKNN(verbose=True)

        # 配置模型（使用w16a16i，更高精度）
        print(f"\n配置RKNN模型...")
        ret = rknn.config(
            mean_values=[[127.5, 127.5, 127.5]],
            std_values=[[127.5, 127.5, 127.5]],
            target_platform=target_platform,
            quantized_dtype='w16a16i',  # INT16，比INT8精度高
            optimization_level=3
        )

        if ret != 0:
            print(f"✗ RKNN配置失败")
            return False

        print(f"✓ RKNN配置成功")

        # 加载ONNX模型
        print(f"\n加载ONNX模型...")
        ret = rknn.load_onnx(model=onnx_path)

        if ret != 0:
            print(f"✗ 加载ONNX模型失败")
            return False

        print(f"✓ ONNX模型加载成功")

        # 构建RKNN模型
        print(f"\n构建RKNN模型...")
        ret = rknn.build(do_quantization=False)  # 不量化

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
    """转换不量化版本"""
    print("="*60)
    print("转换RKNN模型（不量化 FP16版本）")
    print("="*60)

    # 查找ONNX目录
    possible_dirs = ["tool/models/onnx", "models/onnx", "../tool/models/onnx"]
    onnx_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            onnx_dir = d
            break

    if not onnx_dir:
        print("✗ 未找到ONNX目录")
        print(f"  查找路径: {possible_dirs}")
        return 1

    rknn_dir = os.path.join(os.path.dirname(onnx_dir), "rknn_fp16")
    os.makedirs(rknn_dir, exist_ok=True)

    print(f"ONNX目录: {onnx_dir}")
    print(f"RKNN输出: {rknn_dir}")

    # 只转换检测模型测试
    onnx_path = os.path.join(onnx_dir, "ch_PP-OCRv4_det_fixed.onnx")
    rknn_path = os.path.join(rknn_dir, "ch_PP-OCRv4_det_rk3576_fp16.rknn")

    print(f"\n转换检测模型（FP16，不量化）...")
    success = convert_onnx_to_rknn_fp16(
        onnx_path=onnx_path,
        output_path=rknn_path,
        model_name="检测模型 (FP16)",
        target_platform='rk3576'
    )

    if success:
        print(f"\n✓ 转换成功: {rknn_path}")
        print(f"\n将此模型传到开发板测试:")
        print(f"  scp {rknn_path} rk@rk3576-ubuntu:~/ai_ocr/rknn/")
        return 0
    else:
        print(f"\n✗ 转换失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
