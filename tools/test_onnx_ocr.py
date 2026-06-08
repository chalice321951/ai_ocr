"""
完整ONNX OCR测试 - 检测 + 识别
"""
import cv2
import numpy as np
import onnxruntime as ort
import sys

def decode_ctc(preds, char_dict):
    """CTC解码"""
    indices = np.argmax(preds, axis=1)
    chars = []
    last_idx = 0
    for idx in indices:
        if idx != 0 and idx != last_idx and idx < len(char_dict):  # 添加索引检查
            chars.append(char_dict[idx])
        last_idx = idx
    return ''.join(chars)

def test_onnx_ocr(image_path, det_onnx, rec_onnx, dict_path):
    """完整OCR测试"""

    print("="*60)
    print("ONNX完整OCR测试 (检测+识别)")
    print("="*60)

    # 加载字典
    print(f"\n加载字典: {dict_path}")
    with open(dict_path, 'r', encoding='utf-8') as f:
        char_dict = ['blank'] + [line.strip() for line in f]
    print(f"  字符数: {len(char_dict)}")

    # 加载图片
    print(f"\n加载图片: {image_path}")
    image = cv2.imread(image_path)
    print(f"  图片shape: {image.shape}")

    # === 1. 文字检测 ===
    print(f"\n=== 文字检测 ===")
    det_session = ort.InferenceSession(det_onnx)
    det_input_name = det_session.get_inputs()[0].name

    h, w = image.shape[:2]
    ratio_h = 640 / h
    ratio_w = 640 / w

    # 预处理
    resized = cv2.resize(image, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    det_input = normalized.transpose(2, 0, 1)[np.newaxis, ...]

    # 推理
    det_output = det_session.run(None, {det_input_name: det_input})[0]
    pred = det_output[0, 0, :, :]

    print(f"  检测输出范围: [{pred.min():.3f}, {pred.max():.3f}]")

    # 后处理
    binary = (pred > 0.3).astype(np.uint8)  # 二值化阈值
    kernel = np.ones((2, 2), np.uint8)  # 减小kernel
    binary = cv2.dilate(binary, kernel, iterations=1)  # 减少膨胀次数

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 使用EXTERNAL

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # 提高最小面积阈值
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)

        # 计算框内平均得分（添加置信度过滤）
        mask = np.zeros_like(pred, dtype=np.uint8)
        cv2.fillPoly(mask, [box.astype(np.int32)], 1)
        score = (pred * mask).sum() / (mask.sum() + 1e-6)

        if score < 0.5:  # 添加置信度过滤
            continue

        # 映射回原图
        box[:, 0] = box[:, 0] / ratio_w
        box[:, 1] = box[:, 1] / ratio_h

        boxes.append(box)

    print(f"  检测到 {len(boxes)} 个文字区域")

    if len(boxes) == 0:
        print("\n✗ 未检测到文字")
        return

    # === 2. 文字识别 ===
    print(f"\n=== 文字识别 ===")
    rec_session = ort.InferenceSession(rec_onnx)
    rec_input_name = rec_session.get_inputs()[0].name

    results = []
    for idx, box in enumerate(boxes):
        # 裁剪ROI
        box_int = box.astype(np.int32)
        x_min = max(0, int(box[:, 0].min()))
        y_min = max(0, int(box[:, 1].min()))
        x_max = min(image.shape[1], int(box[:, 0].max()))
        y_max = min(image.shape[0], int(box[:, 1].max()))

        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue

        # 预处理识别
        roi_h, roi_w = roi.shape[:2]
        ratio = 48 / roi_h
        new_w = int(roi_w * ratio)

        resized_roi = cv2.resize(roi, (new_w, 48))

        if new_w > 320:
            resized_roi = resized_roi[:, :320, :]
        elif new_w < 320:
            pad_w = 320 - new_w
            resized_roi = cv2.copyMakeBorder(resized_roi, 0, 0, 0, pad_w,
                                            cv2.BORDER_CONSTANT, value=(255, 255, 255))

        rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
        normalized_roi = (rgb_roi.astype(np.float32) / 255.0 - 0.5) / 0.5
        rec_input = normalized_roi.transpose(2, 0, 1)[np.newaxis, ...]

        # 识别
        rec_output = rec_session.run(None, {rec_input_name: rec_input})[0]

        # CTC解码
        text = decode_ctc(rec_output[0], char_dict)

        results.append({
            'box': box,
            'text': text,
            'position': (x_min, y_min)
        })

        print(f"  区域 {idx+1}: {text}")

    # === 3. 可视化 ===
    print(f"\n=== 保存可视化结果 ===")
    vis_image = image.copy()

    for result in results:
        box = result['box'].astype(np.int32)
        cv2.polylines(vis_image, [box], True, (0, 255, 0), 2)

        # 绘制文字（OpenCV不支持中文，所以只画框）
        x, y = result['position']
        cv2.putText(vis_image, result['text'][:5], (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    output_path = "onnx_ocr_result.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"  可视化结果: {output_path}")

    # === 4. 总结 ===
    print(f"\n" + "="*60)
    print(f"识别结果汇总:")
    print(f"="*60)
    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result['text']}")

    print(f"\n✓ ONNX OCR测试完成")
    print(f"  检测: {len(boxes)} 个区域")
    print(f"  识别: {len(results)} 条文字")
    print(f"="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_onnx_ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    det_onnx = "tool/models/onnx/ch_PP-OCRv4_det_fixed.onnx"
    rec_onnx = "tool/models/onnx/ch_PP-OCRv4_rec_fixed.onnx"
    dict_path = "tool/models/ppocr_keys_v1.txt"

    test_onnx_ocr(image_path, det_onnx, rec_onnx, dict_path)
