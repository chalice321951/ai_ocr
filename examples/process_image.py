"""
示例：处理单张图片
"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.ocr_system import OCRSystem
from src.result_formatter import ResultFormatter


def main():
    """处理单张图片的示例"""
    
    # 1. 创建配置（使用默认配置）
    config = Config()
    config.use_gpu = True  # 使用GPU加速
    config.confidence_threshold = 0.5  # 置信度阈值
    
    # 2. 初始化OCR系统
    print("初始化OCR系统...")
    ocr = OCRSystem(config)
    
    # 3. 处理图片
    image_path = "test_image.jpg"  # 替换为你的图片路径
    
    if not os.path.exists(image_path):
        print(f"错误：图片文件不存在: {image_path}")
        print("请将图片放在当前目录，或修改image_path变量")
        return
    
    print(f"\n处理图片: {image_path}")
    result = ocr.process_image(image_path)
    
    # 4. 打印结果
    print(f"\n识别结果:")
    print(f"  处理时间: {result.processing_time:.2f}秒")
    print(f"  识别文字数: {len(result.results)}")
    print()
    
    for i, r in enumerate(result.results, 1):
        print(f"  {i}. 文字: {r.text}")
        print(f"     置信度: {r.confidence:.2f}")
        print(f"     位置: ({r.bbox.x}, {r.bbox.y}), "
              f"大小: {r.bbox.width}x{r.bbox.height}")
        print()
    
    # 5. 保存JSON结果
    json_output = ResultFormatter.format_ocr_result(result, 'json')
    with open('result.json', 'w', encoding='utf-8') as f:
        f.write(json_output)
    print("JSON结果已保存到: result.json")
    
    # 6. 可视化结果（可选）
    try:
        import cv2
        image = cv2.imread(image_path)
        vis_image = ResultFormatter.visualize(image, result.results)
        cv2.imwrite('result_vis.jpg', vis_image)
        print("可视化结果已保存到: result_vis.jpg")
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == '__main__':
    main()
