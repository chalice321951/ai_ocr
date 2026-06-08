"""导出PaddleOCR预训练模型"""
import os
import sys
from pathlib import Path

def export_models():
    """导出检测模型和识别模型"""

    # 创建输出目录
    os.makedirs("models/paddle_models", exist_ok=True)

    print("=" * 60)
    print("初始化PaddleOCR...")
    print("=" * 60)

    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=False,  # 导出时不需要GPU
            show_log=True
        )

        print("\n" + "=" * 60)
        print("✓ PaddleOCR初始化成功")
        print("=" * 60)

        # 获取用户目录
        home = Path.home()
        paddleocr_dir = home / ".paddleocr" / "whl"

        print(f"\nPaddleOCR模型已下载到:")
        print(f"{paddleocr_dir}")
        print("\n检测模型路径:")
        print(f"{paddleocr_dir / 'det' / 'ch' / 'ch_PP-OCRv4_det_infer'}")
        print("\n识别模型路径:")
        print(f"{paddleocr_dir / 'rec' / 'ch' / 'ch_PP-OCRv4_rec_infer'}")
        print("\n方向分类器路径:")
        print(f"{paddleocr_dir / 'cls' / 'ch_ppocr_mobile_v2.0_cls_infer'}")
        print("=" * 60)

        # 检查测试图片
        test_images = [
            "test_image.jpg",
            "examples/test_subway.jpg",
        ]

        test_image = None
        for img in test_images:
            if os.path.exists(img):
                test_image = img
                break

        if test_image:
            print(f"\n测试OCR推理（使用图片: {test_image}）...")
            try:
                result = ocr.ocr(test_image, cls=True)
                if result and result[0]:
                    print(f"✓ OCR推理测试成功")
                    print(f"  检测到 {len(result[0])} 个文字区域")
                    for idx, line in enumerate(result[0][:3]):  # 只显示前3个
                        print(f"  - 文字{idx+1}: {line[1][0]} (置信度: {line[1][1]:.3f})")
                    if len(result[0]) > 3:
                        print(f"  - ... 共{len(result[0])}个结果")
                else:
                    print("⚠ 未检测到文字")
            except Exception as e:
                print(f"✗ OCR推理测试失败: {e}")
        else:
            print("\n⚠ 未找到测试图片，跳过推理测试")
            print("  请准备一张包含中文的测试图片命名为 test_image.jpg")

        print("\n" + "=" * 60)
        print("下一步: 将PaddleOCR模型转换为ONNX格式")
        print("运行: python tools/convert_to_onnx.py")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = export_models()
    sys.exit(0 if success else 1)
