"""修复ONNX模型的动态shape为固定shape"""
import onnx
from onnx import shape_inference
import sys
from pathlib import Path

def fix_dynamic_shape(onnx_path, output_path, input_shape):
    """
    将ONNX模型的动态输入shape固定为指定shape

    Args:
        onnx_path: 输入ONNX模型路径
        output_path: 输出ONNX模型路径
        input_shape: 固定输入shape，例如 [1, 3, 640, 640]
    """
    print(f"\n修复 {Path(onnx_path).name}")
    print(f"  输入: {onnx_path}")
    print(f"  输出: {output_path}")
    print(f"  目标shape: {input_shape}")

    try:
        # 加载模型
        model = onnx.load(onnx_path)

        # 获取输入节点
        graph = model.graph
        input_tensor = graph.input[0]

        print(f"  原始输入名称: {input_tensor.name}")
        print(f"  原始shape: {[d.dim_value if d.dim_value != 0 else d.dim_param for d in input_tensor.type.tensor_type.shape.dim]}")

        # 修改输入shape
        for i, dim in enumerate(input_shape):
            input_tensor.type.tensor_type.shape.dim[i].dim_value = dim
            # 清除动态维度名称
            input_tensor.type.tensor_type.shape.dim[i].ClearField('dim_param')

        print(f"  修改后shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")

        # 形状推理（传播固定shape到整个图）
        print(f"  执行形状推理...")
        model = shape_inference.infer_shapes(model)

        # 保存模型
        onnx.save(model, output_path)

        # 验证
        onnx.checker.check_model(output_path)

        print(f"✓ 修复成功！")
        return True

    except Exception as e:
        print(f"✗ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("ONNX动态shape修复工具")
    print("="*60)

    # 定义要修复的模型（支持多个可能的路径）
    possible_dirs = ["tool/models/onnx", "models/onnx", "../tool/models/onnx"]

    onnx_dir = None
    for dir_path in possible_dirs:
        if Path(dir_path).exists():
            onnx_dir = dir_path
            print(f"✓ 找到ONNX目录: {onnx_dir}")
            break

    if not onnx_dir:
        print(f"✗ 未找到ONNX模型目录")
        print(f"  查找路径: {possible_dirs}")
        return 1

    models = {
        "检测模型": {
            "input": f"{onnx_dir}/ch_PP-OCRv4_det.onnx",
            "output": f"{onnx_dir}/ch_PP-OCRv4_det_fixed.onnx",
            "shape": [1, 3, 640, 640]
        },
        "识别模型": {
            "input": f"{onnx_dir}/ch_PP-OCRv4_rec.onnx",
            "output": f"{onnx_dir}/ch_PP-OCRv4_rec_fixed.onnx",
            "shape": [1, 3, 48, 320]
        },
        "方向分类器": {
            "input": f"{onnx_dir}/ch_ppocr_mobile_v2.0_cls.onnx",
            "output": f"{onnx_dir}/ch_ppocr_mobile_v2.0_cls_fixed.onnx",
            "shape": [1, 3, 48, 192]
        }
    }

    results = {}
    for name, config in models.items():
        success = fix_dynamic_shape(
            config["input"],
            config["output"],
            config["shape"]
        )
        results[name] = success

    # 总结
    print("\n" + "="*60)
    print("修复总结")
    print("="*60)
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{status} - {name}")

    if all(results.values()):
        print("\n✓ 所有模型已修复！")
        print("\n现在可以进行RKNN转换了")
        print("注意：转换时使用 *_fixed.onnx 文件")
        return 0
    else:
        print("\n✗ 部分模型修复失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
