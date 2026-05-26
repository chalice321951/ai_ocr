"""
命令行接口模块
"""
import argparse
import sys
import os
import cv2

from .config import Config
from .ocr_system import OCRSystem
from .result_formatter import ResultFormatter
from .logger import get_logger

logger = get_logger(__name__)


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='OCR文字识别系统 - 支持图片、视频和数据流输入',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 识别单张图片
  python -m src.cli -i image.jpg
  
  # 识别图片并可视化
  python -m src.cli -i image.jpg --visualize -o result.jpg
  
  # 识别视频
  python -m src.cli -v video.mp4 --frame-interval 30
  
  # 使用自定义配置
  python -m src.cli -i image.jpg -c config/config.yaml
  
  # 指定输出格式
  python -m src.cli -i image.jpg --format json -o result.json
        """
    )
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-i', '--image',
        type=str,
        help='输入图片路径'
    )
    input_group.add_argument(
        '-v', '--video',
        type=str,
        help='输入视频路径'
    )
    input_group.add_argument(
        '-s', '--stream',
        type=str,
        help='数据流源（摄像头索引或RTSP URL）'
    )
    input_group.add_argument(
        '-b', '--batch',
        type=str,
        nargs='+',
        help='批量处理多张图片'
    )
    
    # 配置参数
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='配置文件路径（默认使用内置配置）'
    )
    
    # 输出参数
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件路径'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'xml', 'csv'],
        default='json',
        help='输出格式（默认: json）'
    )
    
    # 可视化参数
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='在图像上可视化识别结果'
    )
    
    # 视频处理参数
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=1,
        help='视频帧间隔（默认: 1，处理每一帧）'
    )
    
    # 数据流参数
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='数据流最大处理帧数（默认: 无限制）'
    )
    
    # GPU参数
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='禁用GPU加速'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    try:
        # 加载配置
        if args.config:
            config = Config(args.config)
        else:
            config = Config()
        
        # 如果指定了--no-gpu，覆盖配置
        if args.no_gpu:
            config.use_gpu = False
        
        # 如果指定了输出格式，覆盖配置
        config.output_format = args.format
        
        # 初始化OCR系统
        logger.info("初始化OCR系统...")
        ocr_system = OCRSystem(config)
        
        # 处理输入
        if args.image:
            # 处理单张图片
            result = process_image(ocr_system, args.image, args)
            
        elif args.video:
            # 处理视频
            results = process_video(ocr_system, args.video, args)
            
        elif args.stream:
            # 处理数据流
            process_stream(ocr_system, args.stream, args)
            
        elif args.batch:
            # 批量处理
            results = process_batch(ocr_system, args.batch, args)
        
        logger.info("处理完成！")
        
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
        sys.exit(1)


def process_image(ocr_system: OCRSystem, image_path: str, args) -> None:
    """处理单张图片"""
    logger.info(f"处理图片: {image_path}")
    
    # 执行OCR
    result = ocr_system.process_image(image_path)
    
    # 输出结果
    output_result(result, args)
    
    # 可视化
    if args.visualize:
        visualize_result(image_path, result, args)
    
    return result


def process_video(ocr_system: OCRSystem, video_path: str, args) -> list:
    """处理视频"""
    logger.info(f"处理视频: {video_path}")
    
    # 执行OCR
    results = ocr_system.process_video(video_path, args.frame_interval)
    
    # 输出结果
    if args.output:
        output_video_results(results, args)
    else:
        # 打印摘要
        total_texts = sum(len(r.results) for r in results)
        logger.info(f"视频处理完成: {len(results)} 帧, 共识别 {total_texts} 个文字")
    
    return results


def process_stream(ocr_system: OCRSystem, stream_source: str, args) -> None:
    """处理数据流"""
    # 尝试将stream_source转换为整数（摄像头索引）
    try:
        stream_source = int(stream_source)
    except ValueError:
        pass  # 保持为字符串（RTSP URL等）
    
    logger.info(f"处理数据流: {stream_source}")
    
    # 执行OCR
    for i, result in enumerate(ocr_system.process_stream(stream_source, args.max_frames)):
        logger.info(f"帧 {i+1}: 识别到 {len(result.results)} 个文字")
        
        # 打印识别结果
        for r in result.results:
            print(f"  {r.text} (置信度: {r.confidence:.2f})")


def process_batch(ocr_system: OCRSystem, image_paths: list, args) -> list:
    """批量处理图片"""
    logger.info(f"批量处理 {len(image_paths)} 张图片")
    
    # 执行OCR
    results = ocr_system.batch_process(image_paths)
    
    # 输出结果
    if args.output:
        output_batch_results(results, args)
    else:
        # 打印摘要
        for i, result in enumerate(results):
            logger.info(f"图片 {i+1}: {len(result.results)} 个文字")
    
    return results


def output_result(result, args) -> None:
    """输出单个OCR结果"""
    # 格式化结果
    formatted = ResultFormatter.format_ocr_result(result, args.format)
    
    if args.output:
        # 写入文件
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted)
        logger.info(f"结果已保存到: {args.output}")
    else:
        # 打印到控制台
        print(formatted)


def output_video_results(results: list, args) -> None:
    """输出视频OCR结果"""
    import json
    
    # 将所有结果转换为字典
    data = {
        'total_frames': len(results),
        'frames': [r.to_dict() for r in results]
    }
    
    formatted = json.dumps(data, ensure_ascii=False, indent=2)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(formatted)
    logger.info(f"视频结果已保存到: {args.output}")


def output_batch_results(results: list, args) -> None:
    """输出批量处理结果"""
    import json
    
    # 将所有结果转换为字典
    data = {
        'total_images': len(results),
        'images': [r.to_dict() for r in results]
    }
    
    formatted = json.dumps(data, ensure_ascii=False, indent=2)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(formatted)
    logger.info(f"批量处理结果已保存到: {args.output}")


def visualize_result(image_path: str, result, args) -> None:
    """可视化OCR结果"""
    # 加载原图
    image = cv2.imread(image_path)
    
    if image is None:
        logger.error(f"无法加载图片: {image_path}")
        return
    
    # 可视化
    vis_image = ResultFormatter.visualize(image, result.results)
    
    # 保存或显示
    if args.output:
        # 如果指定了输出文件，保存可视化结果
        output_path = args.output
        if not output_path.lower().endswith(('.jpg', '.png', '.bmp')):
            # 如果输出文件不是图片格式，添加后缀
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_vis.jpg"
        
        cv2.imwrite(output_path, vis_image)
        logger.info(f"可视化结果已保存到: {output_path}")
    else:
        # 显示图片
        cv2.imshow('OCR Result', vis_image)
        logger.info("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
