"""
使用paddle2onnx官方方法重新转换并固定shape
参考：https://blog.csdn.net/m0_60657960/article/details/145679346
"""
import os
import subprocess
import sys

def convert_models():
    """重新转换模型，使用正确的输入尺寸"""

    print("="*60)
    print("使用paddle2onnx官方方法重新转换模型")
    print("="*60)

    # PaddleOCR模型目录
    home = os.path.expanduser("~")
    paddleocr_dir = os.path.join(home, ".paddleocr", "whl")

    models = {
        "检测模型": {
            "model_dir": os.path.join(paddleocr_dir, "det/ch/ch_PP-OCRv4_det_infer"),
            "output": "tool/models/onnx_960/ch_PP-OCRv4_det.onnx",
            "input_shape": "{'x':[1,3,960,960]}"  # 注意：960×960
        },
        "识别模型": {
            "model_dir": os.path.join(paddleocr_dir, "rec/ch/ch_PP-OCRv4_rec_infer"),
            "output": "tool/models/onnx_960/ch_PP-OCRv4_rec.onnx",
            "input_shape": "{'x':[1,3,48,320]}"
        },
        "方向分类器": {
            "model_dir": os.path.join(paddleocr_dir, "cls/ch_ppocr_mobile_v2.0_cls_infer"),
            "output": "tool/models/onnx_960/ch_ppocr_mobile_v2.0_cls.onnx",
            "input_shape": "{'x':[1,3,48,192]}"
        }
    }

    # 创建输出目录
    os.makedirs("tool/models/onnx_960", exist_ok=True)

    for name, config in models.items():
        print(f"\n{'='*60}")
        print(f"转换 {name}")
        print(f"{'='*60}")

        model_dir = config["model_dir"]
        output = config["output"]
        input_shape = config["input_shape"]

        # 步骤1：paddle2onnx转换
        print(f"\n步骤1：Paddle → ONNX")
        cmd1 = [
            "paddle2onnx",
            "--model_dir", model_dir,
            "--model_filename", "inference.pdmodel",
            "--params_filename", "inference.pdiparams",
            "--save_file", output,
            "--opset_version", "11"
        ]

        print(f"命令: {' '.join(cmd1)}")
        result = subprocess.run(cmd1, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ 转换失败: {result.stderr}")
            continue

        print(f"✓ ONNX转换成功")

        # 步骤2：固定shape
        print(f"\n步骤2：固定输入shape为 {input_shape}")
        cmd2 = [
            "python3", "-m", "paddle2onnx.optimize",
            "--input_model", output,
            "--output_model", output,
            "--input_shape_dict", input_shape
        ]

        print(f"命令: {' '.join(cmd2)}")
        result = subprocess.run(cmd2, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ 固定shape失败: {result.stderr}")
            continue

        print(f"✓ Shape固定成功")

        # 验证
        file_size = os.path.getsize(output) / (1024 * 1024)
        print(f"✓ 模型文件: {output} ({file_size:.2f} MB)")

    print(f"\n{'='*60}")
    print("转换完成！")
    print(f"{'='*60}")
    print("\n模型保存在: tool/models/onnx_960/")
    print("\n注意：检测模型现在是960×960，需要更新转换脚本")
    print("\n下一步：")
    print("  python tool/convert_to_rknn_960.py")

if __name__ == "__main__":
    convert_models()
