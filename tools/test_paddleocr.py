"""
测试原始PaddleOCR（GPU/CPU版本）
对比转换前后的效果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddleocr import PaddleOCR
import cv2

def test_paddleocr(image_path):
    """测试原始PaddleOCR"""

    print("="*60)
    print("原始PaddleOCR测试")
    print("="*60)

    # 加载图片
    print(f"\n加载图片: {image_path}")
    image = cv2.imread(image_path)
    print(f"  图片shape: {image.shape}")

    # 初始化PaddleOCR
    print(f"\n初始化PaddleOCR (CPU模式)...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        use_gpu=False,  # 服务器上用CPU测试
        show_log=True
    )

    print(f"\n开始OCR识别...")
    result = ocr.ocr(image_path, cls=True)

    # 显示结果
    print(f"\n{'='*60}")
    print(f"识别结果:")
    print(f"{'='*60}")

    if result and result[0]:
        print(f"检测到 {len(result[0])} 个文字区域\n")

        for idx, line in enumerate(result[0], 1):
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]

            # 计算box位置
            x = int(box[0][0])
            y = int(box[0][1])

            print(f"{idx:2d}. [{x:4d},{y:4d}] {text} (置信度: {confidence:.3f})")
    else:
        print("✗ 未检测到文字")

    # 保存可视化
    print(f"\n{'='*60}")
    print(f"保存可视化结果")
    print(f"{'='*60}")

    if result and result[0]:
        from PIL import Image, ImageDraw, ImageFont

        # 转为PIL格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载中文字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()

        for line in result[0]:
            box = line[0]
            text = line[1][0]

            # 画框
            box_int = [(int(p[0]), int(p[1])) for p in box]
            draw.polygon(box_int, outline=(0, 255, 0), width=2)

            # 画文字
            draw.text((int(box[0][0]), int(box[0][1]) - 25), text, fill=(255, 0, 0), font=font)

        # 保存
        output_path = "paddleocr_result.jpg"
        pil_img.save(output_path)
        print(f"  可视化结果: {output_path}")

    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_paddleocr.py <image_path>")
        print("\n示例:")
        print("  python test_paddleocr.py /home/admin123/wjj/ocr/images/calib_0000_original.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    test_paddleocr(image_path)
