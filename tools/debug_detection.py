"""
调试RKNN检测模型
输出中间结果，分析为什么检测不到文字
"""
import cv2
import numpy as np
import sys
from rknnlite.api import RKNNLite

def debug_detection(image_path, model_path):
    """调试检测过程"""

    print("="*60)
    print("RKNN检测模型调试")
    print("="*60)

    # 1. 加载图片
    print(f"\n1. 加载图片: {image_path}")
    image = cv2.imread(image_path)
    print(f"   图片shape: {image.shape}")
    print(f"   图片dtype: {image.dtype}")
    print(f"   图片范围: [{image.min()}, {image.max()}]")

    # 2. 加载模型
    print(f"\n2. 加载模型: {model_path}")
    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"   ✗ 加载失败")
        return

    ret = rknn.init_runtime(core_mask=3)
    if ret != 0:
        print(f"   ✗ 初始化失败")
        return

    print(f"   ✓ 模型加载成功")

    # 3. 预处理 - 测试多种归一化方式
    print(f"\n3. 测试不同的预处理方式")

    h, w = image.shape[:2]
    input_size = (640, 640)

    # Resize
    resized = cv2.resize(image, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 测试多种归一化
    preprocess_methods = {
        "方法1: /255.0": rgb.astype(np.float32) / 255.0,
        "方法2: (x/255.0 - 0.5)/0.5": (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5,
        "方法3: (x - 127.5)/127.5": (rgb.astype(np.float32) - 127.5) / 127.5,
        "方法4: x/255.0, mean=[0.485,0.456,0.406]": None  # 稍后计算
    }

    # 方法4
    normalized = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    preprocess_methods["方法4: x/255.0, mean=[0.485,0.456,0.406]"] = (normalized - mean) / std

    for method_name, normalized in preprocess_methods.items():
        print(f"\n   测试: {method_name}")
        print(f"   范围: [{normalized.min():.3f}, {normalized.max():.3f}]")

        # CHW格式
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)

        # 推理
        outputs = rknn.inference(inputs=[batched])
        output = outputs[0]

        print(f"   输出shape: {output.shape}")
        print(f"   输出范围: [{output.min():.6f}, {output.max():.6f}]")
        print(f"   输出均值: {output.mean():.6f}")

        # 统计大于阈值的像素
        pred = output[0, 0, :, :]
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            count = (pred > thresh).sum()
            print(f"   像素 > {thresh}: {count} ({count/(pred.size)*100:.2f}%)")

        # 保存热力图
        heatmap = (pred * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        output_path = f"debug_heatmap_{method_name.split(':')[0].replace(' ', '_')}.jpg"
        cv2.imwrite(output_path, heatmap_color)
        print(f"   热力图已保存: {output_path}")

    rknn.release()

    print("\n" + "="*60)
    print("调试完成")
    print("="*60)
    print("\n请检查生成的热力图，看哪种预处理方式效果最好")
    print("热力图中亮的区域表示模型认为有文字的地方")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python debug_detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "/home/rk/ai_ocr/rknn/ch_PP-OCRv4_det_rk3576.rknn"

    debug_detection(image_path, model_path)
