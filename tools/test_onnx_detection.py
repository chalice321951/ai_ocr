"""
测试ONNX模型是否能正常检测
在服务器上验证ONNX模型本身没问题
"""
import cv2
import numpy as np
import onnxruntime as ort
import sys

def test_onnx_detection(image_path, onnx_path):
    """测试ONNX检测模型"""

    print("="*60)
    print("ONNX检测模型测试")
    print("="*60)

    # 1. 加载图片
    print(f"\n1. 加载图片: {image_path}")
    image = cv2.imread(image_path)
    print(f"   图片shape: {image.shape}")

    # 2. 加载ONNX模型
    print(f"\n2. 加载ONNX模型: {onnx_path}")
    session = ort.InferenceSession(onnx_path)

    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name

    print(f"   输入名称: {input_name}")
    print(f"   输入shape: {input_shape}")
    print(f"   输出名称: {output_name}")

    # 3. 预处理
    print(f"\n3. 预处理图片")
    h, w = image.shape[:2]

    # Resize到640x640
    resized = cv2.resize(image, (640, 640))

    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 归一化到[-1, 1] (PaddleOCR标准)
    normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5

    # HWC to CHW
    transposed = normalized.transpose(2, 0, 1)

    # Add batch
    batched = np.expand_dims(transposed, axis=0)

    print(f"   预处理后shape: {batched.shape}")
    print(f"   数据范围: [{batched.min():.3f}, {batched.max():.3f}]")

    # 4. 推理
    print(f"\n4. ONNX推理")
    outputs = session.run([output_name], {input_name: batched})
    output = outputs[0]

    print(f"   输出shape: {output.shape}")
    print(f"   输出范围: [{output.min():.6f}, {output.max():.6f}]")
    print(f"   输出均值: {output.mean():.6f}")

    # 5. 分析输出
    print(f"\n5. 分析检测结果")
    pred = output[0, 0, :, :]

    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        count = (pred > thresh).sum()
        percentage = count / pred.size * 100
        print(f"   像素 > {thresh}: {count} ({percentage:.2f}%)")

    # 6. 保存热力图
    print(f"\n6. 保存热力图")
    heatmap = (pred * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    output_path = "onnx_heatmap.jpg"
    cv2.imwrite(output_path, heatmap_color)
    print(f"   热力图: {output_path}")

    # 7. 简单后处理
    print(f"\n7. 简单检测文字区域")
    binary = (pred > 0.3).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(f"   检测到 {len(contours)} 个轮廓")

    # 过滤太小的轮廓
    valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
    print(f"   有效区域: {len(valid_contours)} 个")

    print("\n" + "="*60)
    if len(valid_contours) > 0:
        print("✓ ONNX模型检测正常！")
        print("  模型本身没问题，是RKNN量化过程出了问题")
    else:
        print("✗ ONNX模型也检测不到")
        print("  可能是ONNX转换或模型本身的问题")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_onnx_detection.py <image_path> [onnx_path]")
        print("\n示例:")
        print("  python test_onnx_detection.py test_image.jpg")
        print("  python test_onnx_detection.py test_image.jpg tool/models/onnx/ch_PP-OCRv4_det_fixed.onnx")
        sys.exit(1)

    image_path = sys.argv[1]
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else "tool/models/onnx/ch_PP-OCRv4_det_fixed.onnx"

    test_onnx_detection(image_path, onnx_path)
