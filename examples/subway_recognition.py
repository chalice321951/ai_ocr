"""
示例：地铁站名识别
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.ocr_system import OCRSystem
from src.subway_processor import SubwayProcessor


def main():
    """地铁站名识别的示例"""
    
    # 1. 创建配置
    config = Config()
    config.use_gpu = True
    config.confidence_threshold = 0.5
    config.rec_lang = "ch"  # 中英文混合识别
    
    # 2. 初始化OCR系统
    print("初始化OCR系统...")
    ocr = OCRSystem(config)
    
    # 3. 处理地铁线路图
    image_path = "subway_map.jpg"  # 替换为你的地铁线路图路径
    
    if not os.path.exists(image_path):
        print(f"错误：图片文件不存在: {image_path}")
        print("请将地铁线路图放在当前目录，或修改image_path变量")
        return
    
    print(f"\n处理地铁线路图: {image_path}")
    result = ocr.process_image(image_path)
    
    print(f"  识别到 {len(result.results)} 个文字区域")
    
    # 4. 使用地铁站名处理器解析站名
    print("\n解析地铁站名...")
    processor = SubwayProcessor(confidence_threshold=0.5)
    station_names = processor.parse_station_names(result.results)
    
    # 5. 打印站名
    print(f"\n识别到 {len(station_names)} 个站名:")
    print("=" * 60)
    
    for i, station in enumerate(station_names, 1):
        print(f"{i}. {station.chinese:10s} {station.english:30s} "
              f"(置信度: {station.confidence:.2f})")
    
    print("=" * 60)
    
    # 6. 按关键词过滤站名（可选）
    keywords = ["河街", "Street"]  # 示例关键词
    filtered_stations = processor.filter_by_keywords(station_names, keywords)
    
    if filtered_stations:
        print(f"\n包含关键词 {keywords} 的站名:")
        for station in filtered_stations:
            print(f"  - {station.chinese} {station.english}")
    
    # 7. 保存结果
    import json
    output_data = {
        'image_path': image_path,
        'total_stations': len(station_names),
        'stations': [
            {
                'chinese': s.chinese,
                'english': s.english,
                'confidence': s.confidence,
                'position': {'x': s.position[0], 'y': s.position[1]}
            }
            for s in station_names
        ]
    }
    
    with open('subway_stations.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到: subway_stations.json")
    
    # 8. 可视化结果（可选）
    try:
        import cv2
        from src.result_formatter import ResultFormatter
        
        image = cv2.imread(image_path)
        vis_image = ResultFormatter.visualize(image, result.results)
        cv2.imwrite('subway_result_vis.jpg', vis_image)
        print("可视化结果已保存到: subway_result_vis.jpg")
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == '__main__':
    main()
