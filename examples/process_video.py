"""
示例：处理视频文件
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.ocr_system import OCRSystem


def main():
    """处理视频文件的示例"""
    
    # 1. 创建配置
    config = Config()
    config.use_gpu = True
    config.confidence_threshold = 0.6  # 提高阈值以减少误识别
    
    # 2. 初始化OCR系统
    print("初始化OCR系统...")
    ocr = OCRSystem(config)
    
    # 3. 处理视频
    video_path = "test_video.mp4"  # 替换为你的视频路径
    frame_interval = 30  # 每30帧处理一次（约每秒一次，假设30fps）
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        print("请将视频放在当前目录，或修改video_path变量")
        return
    
    print(f"\n处理视频: {video_path}")
    print(f"帧间隔: {frame_interval}")
    
    results = ocr.process_video(video_path, frame_interval)
    
    # 4. 分析结果
    print(f"\n视频处理完成!")
    print(f"  总帧数: {len(results)}")
    
    # 统计识别到文字的帧数
    frames_with_text = sum(1 for r in results if r.results)
    print(f"  识别到文字的帧数: {frames_with_text}")
    
    # 统计总文字数
    total_texts = sum(len(r.results) for r in results)
    print(f"  总识别文字数: {total_texts}")
    
    # 5. 打印每一帧的识别结果
    print("\n每帧识别结果:")
    for i, result in enumerate(results, 1):
        if result.results:
            texts = [r.text for r in result.results]
            print(f"  帧 {i}: {', '.join(texts)}")
    
    # 6. 保存结果到文件
    import json
    output_data = {
        'video_path': video_path,
        'frame_interval': frame_interval,
        'total_frames': len(results),
        'frames_with_text': frames_with_text,
        'total_texts': total_texts,
        'frames': [r.to_dict() for r in results]
    }
    
    with open('video_result.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到: video_result.json")


if __name__ == '__main__':
    main()
