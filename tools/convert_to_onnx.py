"""将PaddleOCR模型转换为ONNX格式"""
import os
import sys
from pathlib import Path
import paddle2onnx

def convert_paddle_to_onnx(model_dir, output_path, model_name, input_shape=None):
    """
    转换PaddlePaddle模型为ONNX格式

    Args:
        model_dir: PaddlePaddle模型目录
        output_path: 输出ONNX文件路径
        model_name: 模型名称（用于日志）
        input_shape: 固定输入shape，例如 [1, 3, 640, 640]
    """
    print(f"\n{'='*60}")
    print(f"转换 {model_name}")
    print(f"{'='*60}")
    print(f"输入目录: {model_dir}")
    print(f"输出文件: {output_path}")
    if input_shape:
        print(f"固定输入: {input_shape}")

    model_file = os.path.join(model_dir, "inference.pdmodel")
    params_file = os.path.join(model_dir, "inference.pdiparams")

    # 检查文件是否存在
    if not os.path.exists(model_file):
        print(f"✗ 模型文件不存在: {model_file}")
        return False
    if not os.path.exists(params_file):
        print(f"✗ 参数文件不存在: {params_file}")
        return False

    try:
        # 构建paddle2onnx转换参数
        convert_params = {
            'model_file': model_file,
            'params_file': params_file,
            'opset_version': 11,
            'save_file': output_path,
            'enable_onnx_checker': True
        }

        # 如果指定了固定输入shape，添加input_shape参数
        if input_shape:
            # paddle2onnx需要的格式：'x:1,3,640,640'
            shape_str = ','.join(map(str, input_shape))
            convert_params['input_shape_dict'] = {'x': input_shape}
            print(f"  设置固定输入shape: x -> {input_shape}")

        # 转换为ONNX
        onnx_model = paddle2onnx.export(**convert_params)

        # 检查转换结果
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✓ 转换成功！")
            print(f"  文件大小: {file_size:.2f} MB")
            return True
        else:
            print(f"✗ 转换失败，输出文件未生成")
            return False

    except Exception as e:
        print(f"✗ 转换出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_onnx_model(onnx_path):
    """验证ONNX模型"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ✓ ONNX模型验证通过")

        # 打印模型信息
        print(f"  输入节点数: {len(model.graph.input)}")
        print(f"  输出节点数: {len(model.graph.output)}")
        if len(model.graph.input) > 0:
            print(f"  输入名称: {model.graph.input[0].name}")
        if len(model.graph.output) > 0:
            print(f"  输出名称: {model.graph.output[0].name}")

        return True
    except Exception as e:
        print(f"  ✗ ONNX模型验证失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("PaddleOCR模型转ONNX工具")
    print("="*60)

    # 获取用户目录
    home = Path.home()
    paddleocr_dir = home / ".paddleocr" / "whl"

    # 定义模型路径和固定输入shape
    models = {
        "检测模型 (ch_PP-OCRv4_det)": {
            "input": paddleocr_dir / "det" / "ch" / "ch_PP-OCRv4_det_infer",
            "output": "models/onnx/ch_PP-OCRv4_det.onnx",
            "input_shape": [1, 3, 640, 640]  # 固定shape用于RKNN
        },
        "识别模型 (ch_PP-OCRv4_rec)": {
            "input": paddleocr_dir / "rec" / "ch" / "ch_PP-OCRv4_rec_infer",
            "output": "models/onnx/ch_PP-OCRv4_rec.onnx",
            "input_shape": [1, 3, 48, 320]
        },
        "方向分类器 (ch_ppocr_mobile_v2.0_cls)": {
            "input": paddleocr_dir / "cls" / "ch_ppocr_mobile_v2.0_cls_infer",
            "output": "models/onnx/ch_ppocr_mobile_v2.0_cls.onnx",
            "input_shape": [1, 3, 48, 192]
        }
    }

    # 创建输出目录
    os.makedirs("models/onnx", exist_ok=True)

    # 转换所有模型
    results = {}
    for model_name, paths in models.items():
        input_dir = str(paths["input"])
        output_path = paths["output"]
        input_shape = paths.get("input_shape")

        # 转换（指定固定shape）
        success = convert_paddle_to_onnx(input_dir, output_path, model_name, input_shape)
        results[model_name] = success

        # 验证
        if success:
            verify_onnx_model(output_path)

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
        print("\n下一步: 将ONNX模型转换为RKNN格式")
        print("运行: python tools/convert_to_rknn.py")
    else:
        print("\n✗ 部分模型转换失败，请检查错误信息")
        return 1

    print("="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
