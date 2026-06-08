"""
RK3576推理测试脚本
测试RKNN模型在开发板上的推理效果
"""
import cv2
import numpy as np
import time
import sys
import os

# 添加父目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rknn_inference import RKNNDetector, RKNNRecognizer


def test_single_image(image_path: str,
                      det_model: str,
                      rec_model: str,
                      dict_path: str):
    """
    测试单张图片

    Args:
        image_path: 图片路径
        det_model: 检测模型路径
        rec_model: 识别模型路径
        dict_path: 字符字典路径
    """
    print("="*60)
    print("RKNN推理测试 - RK3576")
    print("="*60)

    # 1. 加载图片
    print(f"\n加载图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ 无法加载图片: {image_path}")
        return

    print(f"  图片尺寸: {image.shape[1]}x{image.shape[0]}")

    # 2. 初始化检测器
    print(f"\n初始化检测器...")
    start_time = time.time()
    detector = RKNNDetector(det_model, core_mask=3)  # 使用双核
    init_det_time = time.time() - start_time
    print(f"  初始化耗时: {init_det_time:.3f}秒")

    # 3. 初始化识别器
    print(f"\n初始化识别器...")
    start_time = time.time()
    recognizer = RKNNRecognizer(rec_model, dict_path, core_mask=3)
    init_rec_time = time.time() - start_time
    print(f"  初始化耗时: {init_rec_time:.3f}秒")

    # 4. 文字检测
    print(f"\n文字检测...")
    start_time = time.time()
    boxes = detector.detect(image)
    det_time = time.time() - start_time
    print(f"  检测耗时: {det_time*1000:.1f}ms")
    print(f"  检测到 {len(boxes)} 个文字区域")

    if len(boxes) == 0:
        print("✗ 未检测到文字区域")
        return

    # 5. 文字识别
    print(f"\n文字识别...")
    results = []
    total_rec_time = 0

    for idx, box in enumerate(boxes):
        # 裁剪文字区域
        box_int = box.astype(np.int32)
        x_min = max(0, int(box[:, 0].min()))
        y_min = max(0, int(box[:, 1].min()))
        x_max = min(image.shape[1], int(box[:, 0].max()))
        y_max = min(image.shape[0], int(box[:, 1].max()))

        roi = image[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            continue

        # 识别
        start_time = time.time()
        text, confidence = recognizer.recognize(roi)
        rec_time = time.time() - start_time
        total_rec_time += rec_time

        results.append({
            'box': box,
            'text': text,
            'confidence': confidence
        })

        print(f"  区域{idx+1}: '{text}' (置信度: {confidence:.3f}, 耗时: {rec_time*1000:.1f}ms)")

    avg_rec_time = total_rec_time / len(boxes) if boxes else 0

    # 6. 总结
    print(f"\n{'='*60}")
    print("推理性能总结")
    print(f"{'='*60}")
    print(f"检测耗时: {det_time*1000:.1f}ms")
    print(f"识别耗时: {total_rec_time*1000:.1f}ms (平均 {avg_rec_time*1000:.1f}ms/区域)")
    print(f"总耗时: {(det_time + total_rec_time)*1000:.1f}ms")
    print(f"FPS: {1/(det_time + total_rec_time):.1f}")

    # 7. 可视化结果
    vis_image = image.copy()
    for result in results:
        box = result['box'].astype(np.int32)
        cv2.polylines(vis_image, [box], True, (0, 255, 0), 2)

        # 绘制文字（简单版本，RK3576上不支持中文）
        x, y = int(box[0][0]), int(box[0][1]) - 5
        text_label = f"{result['text'][:10]}..."  # 截断显示
        cv2.putText(vis_image, text_label, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 保存结果
    output_path = "test_result_rknn.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"\n✓ 可视化结果已保存: {output_path}")

    # 8. 识别结果
    print(f"\n识别结果:")
    for idx, result in enumerate(results):
        print(f"  {idx+1}. {result['text']} (置信度: {result['confidence']:.3f})")

    print(f"{'='*60}")


if __name__ == "__main__":
    # 默认路径（支持多个可能的位置）
    image_path = "test_image.jpg"

    # 查找模型和字典的路径（优先使用绝对路径）
    possible_model_dirs = [
        "/home/rk/ai_ocr/rknn",
        os.path.expanduser("~/ai_ocr/rknn"),
        "rknn",
        "models/rknn"
    ]

    model_dir = None
    for dir_path in possible_model_dirs:
        full_path = os.path.abspath(dir_path)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            # 检查是否有模型文件
            det_check = os.path.join(full_path, "ch_PP-OCRv4_det_rk3576.rknn")
            if os.path.exists(det_check):
                model_dir = full_path
                print(f"✓ 找到模型目录: {model_dir}")
                break

    if not model_dir:
        print("✗ 未找到模型目录或模型文件")
        print(f"  查找路径: {possible_model_dirs}")
        print(f"\n请确认模型文件在以下位置:")
        print(f"  /home/rk/ai_ocr/rknn/ch_PP-OCRv4_det_rk3576.rknn")
        sys.exit(1)

    det_model = os.path.join(model_dir, "ch_PP-OCRv4_det_rk3576.rknn")
    rec_model = os.path.join(model_dir, "ch_PP-OCRv4_rec_rk3576.rknn")

    # 查找字典文件
    possible_dict_paths = [
        os.path.join(model_dir, "ppocr_keys_v1.txt"),
        "/home/rk/ai_ocr/rknn/ppocr_keys_v1.txt",
        "ppocr_keys_v1.txt"
    ]

    dict_path = None
    for path in possible_dict_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            dict_path = full_path
            print(f"✓ 找到字典文件: {dict_path}")
            break

    if not dict_path:
        print("✗ 未找到字典文件")
        print(f"  查找路径: {possible_dict_paths}")
        sys.exit(1)

    # 命令行参数覆盖
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        det_model = sys.argv[2]
    if len(sys.argv) > 3:
        rec_model = sys.argv[3]
    if len(sys.argv) > 4:
        dict_path = sys.argv[4]

    # 运行测试
    try:
        test_single_image(image_path, det_model, rec_model, dict_path)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
