"""
完整的DB后处理算法
参考PaddleOCR官方实现
"""
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess:
    """DB检测后处理"""

    def __init__(self, thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5):
        """
        Args:
            thresh: 二值化阈值
            box_thresh: 框置信度阈值
            max_candidates: 最大候选框数
            unclip_ratio: 扩展比例
        """
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

    def __call__(self, pred, shape):
        """
        Args:
            pred: 模型输出 (H, W)
            shape: 原图尺寸 (src_h, src_w)

        Returns:
            检测框列表 [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
        """
        pred = pred[:, :, 0]  # (H, W)
        segmentation = pred > self.thresh

        boxes, scores = self.boxes_from_bitmap(pred, segmentation, shape)

        return boxes, scores

    def boxes_from_bitmap(self, pred, bitmap, dest_shape):
        """从概率图提取框"""
        height, width = bitmap.shape
        dest_h, dest_w = dest_shape

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []

        for i in range(num_contours):
            contour = contours[i]
            points, sside = self.get_mini_boxes(contour)

            if sside < self.min_size:
                continue

            # 计算框内平均分数
            score = self.box_score_fast(pred, contour)
            if score < self.box_thresh:
                continue

            # 扩展框
            box = self.unclip(points, self.unclip_ratio)
            if box is None:
                continue

            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            # 映射到原图
            box[:, 0] = np.clip(box[:, 0] / width * dest_w, 0, dest_w)
            box[:, 1] = np.clip(box[:, 1] / height * dest_h, 0, dest_h)

            boxes.append(box)
            scores.append(score)

        return boxes, scores

    def unclip(self, box, unclip_ratio):
        """扩展框"""
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)

        if len(expanded) == 0:
            return None

        expanded = np.array(expanded[0])
        return expanded

    def get_mini_boxes(self, contour):
        """获取最小外接矩形"""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        box = np.array(box)

        # 计算短边
        side = min(bounding_box[1])

        return box, side

    def box_score_fast(self, bitmap, contour):
        """计算框的平均得分"""
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(int(contour[:, 0].min()), 0, w - 1)
        xmax = np.clip(int(contour[:, 0].max()), 0, w - 1)
        ymin = np.clip(int(contour[:, 1].min()), 0, h - 1)
        ymax = np.clip(int(contour[:, 1].max()), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, [contour.astype(np.int32)], 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


if __name__ == "__main__":
    # 测试
    import sys
    import onnxruntime as ort

    if len(sys.argv) < 2:
        print("用法: python db_postprocess.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    det_onnx_path = "tool/models/onnx/ch_PP-OCRv4_det_fixed.onnx"
    rec_onnx_path = "tool/models/onnx/ch_PP-OCRv4_rec_fixed.onnx"
    dict_path = "tool/models/ppocr_keys_v1.txt"

    # 加载字典
    print("加载字典...")
    with open(dict_path, 'r', encoding='utf-8') as f:
        char_dict = ['blank'] + [line.strip() for line in f]
    print(f"  字符数: {len(char_dict)}")

    # 加载图片
    image = cv2.imread(image_path)
    print(f"\n图片shape: {image.shape}")

    # === 检测 ===
    print(f"\n{'='*60}")
    print("文字检测")
    print(f"{'='*60}")

    det_session = ort.InferenceSession(det_onnx_path)
    input_name = det_session.get_inputs()[0].name

    h, w = image.shape[:2]
    resized = cv2.resize(image, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    det_input = normalized.transpose(2, 0, 1)[np.newaxis, ...]

    det_output = det_session.run(None, {input_name: det_input})[0]

    # DB后处理
    post_process = DBPostProcess(thresh=0.3, box_thresh=0.6, unclip_ratio=1.5)
    boxes, scores = post_process(det_output[0].transpose(1, 2, 0), (h, w))

    print(f"检测到 {len(boxes)} 个文字区域")

    if len(boxes) == 0:
        print("✗ 未检测到文字区域")
        sys.exit(0)

    # === 识别 ===
    print(f"\n{'='*60}")
    print("文字识别")
    print(f"{'='*60}")

    rec_session = ort.InferenceSession(rec_onnx_path)
    rec_input_name = rec_session.get_inputs()[0].name

    results = []
    for idx, (box, score) in enumerate(zip(boxes, scores), 1):
        # 裁剪ROI
        box_int = box.astype(np.int32)
        x_min = max(0, int(box[:, 0].min()))
        y_min = max(0, int(box[:, 1].min()))
        x_max = min(w, int(box[:, 0].max()))
        y_max = min(h, int(box[:, 1].max()))

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
        indices = np.argmax(rec_output[0], axis=1)
        chars = []
        last_idx = 0
        for char_idx in indices:
            if char_idx != 0 and char_idx != last_idx and char_idx < len(char_dict):
                chars.append(char_dict[char_idx])
            last_idx = char_idx
        text = ''.join(chars)

        results.append({
            'box': box,
            'text': text,
            'det_score': score
        })

        print(f"区域 {idx}: {text} (检测置信度: {score:.3f})")

    # === 可视化 ===
    print(f"\n{'='*60}")
    print("保存可视化结果")
    print(f"{'='*60}")

    vis_image = image.copy()
    for result in results:
        box_int = result['box'].astype(np.int32)
        cv2.polylines(vis_image, [box_int], True, (0, 255, 0), 2)

    output_path = "db_ocr_result.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"可视化结果: {output_path}")

    # === 总结 ===
    print(f"\n{'='*60}")
    print("识别结果汇总")
    print(f"{'='*60}")
    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result['text']}")

    print(f"\n✓ 完成")
    print(f"{'='*60}")
