"""
OCR文字识别系统 - 主程序
直接运行此文件即可使用OCR功能
所有配置在 config/config.yaml 文件中修改
"""
import sys
import os
import traceback
import cv2

from src.config import Config
from src.ocr_system import OCRSystem
from src.result_formatter import ResultFormatter
from src.subway_processor import SubwayProcessor
from src.logger import setup_logger

# 设置日志
logger = setup_logger("Main", log_level="INFO")


def main():
    """主函数 - 从配置文件读取所有配置"""
    
    logger.info("=" * 50)
    logger.info("OCR文字识别系统启动")
    logger.info("=" * 50)
    
    try:
        # 加载配置文件
        config_path = "config/config.yaml"
        
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            logger.info("请确保 config/config.yaml 文件存在")
            return
        
        logger.info(f"加载配置: {config_path}")
        config = Config(config_path)

        logger.info(f"模式={config.input_mode}  GPU={config.use_gpu}  语言={config.rec_lang}")

        ocr = OCRSystem(config)
        
        # 根据配置的模式执行
        if config.input_mode == "image":
            process_image(ocr, config)
        
        elif config.input_mode == "video":
            process_video(ocr, config)
        
        elif config.input_mode == "batch":
            process_batch(ocr, config)
        
        elif config.input_mode == "stream":
            process_stream(ocr, config)
        
        else:
            logger.error(f"不支持的输入模式: {config.input_mode}")
            logger.info("支持的模式: image, video, batch, stream")
            return
        
        logger.info("\n处理完成！")
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"程序出错: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
        
    finally:
        logger.info("OCR文字识别系统退出")


def process_image(ocr: OCRSystem, config: Config):
    """处理单张图片"""
    image_path = config.image_path

    logger.info(f"处理: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在: {image_path}")
        logger.info(f"请检查配置文件中的 input.image_path 设置")
        return

    # 执行OCR
    result = ocr.process_image(image_path)

    # 打印所有识别结果
    print("\n" + "=" * 60)
    print("全部识别结果:")
    print("=" * 60)
    print(f"处理时间: {result.processing_time:.2f}秒")
    print(f"识别文字数: {len(result.results)}\n")

    for i, r in enumerate(result.results, 1):
        print(f"{i}. {r.text} (置信度: {r.confidence:.2f})")

    print("=" * 60)

    # 过滤站点信息（当前站/下一站）
    print("\n" + "=" * 60)
    print("站点信息过滤结果:")
    print("=" * 60)

    subway_processor = SubwayProcessor()
    station_info = subway_processor.filter_station_info(result.results)

    if station_info:
        for info in station_info:
            print(f"[{info['type']}] {info['station_name']} (置信度: {info['confidence']:.2f})")
    else:
        print("未找到 '当前站：XXX' 或 '下一站：XXX' 格式的内容")

    print("=" * 60)

    # 保存结果
    if config.save_result:
        output_file = f"result.{config.output_format}"
        formatted = ResultFormatter.format_ocr_result(result, config.output_format)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted)
        logger.info(f"结果已保存到: {output_file}")

        # 保存过滤后的站点信息
        import json
        station_output_file = "station_info.json"
        with open(station_output_file, 'w', encoding='utf-8') as f:
            json.dump(station_info, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"站点信息已保存到: {station_output_file}")

    # 可视化
    if config.visualize:
        try:
            image = cv2.imread(image_path)
            vis_image = ResultFormatter.visualize(image, result.results)
            vis_output = "result_vis.jpg"
            cv2.imwrite(vis_output, vis_image)
            logger.info(f"可视化结果已保存到: {vis_output}")
        except Exception as e:
            logger.warning(f"可视化失败: {e}")


def process_video(ocr: OCRSystem, config: Config):
    """处理视频"""
    video_path = config.video_path
    frame_interval = config.frame_interval
    
    logger.info(f"处理视频: {video_path}")
    logger.info(f"帧间隔: {frame_interval}")
    
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        logger.info(f"请检查配置文件中的 input.video_path 设置")
        return
    
    # 执行OCR
    results = ocr.process_video(video_path, frame_interval)
    
    # 打印摘要
    total_texts = sum(len(r.results) for r in results)
    print("\n" + "=" * 60)
    print("视频处理完成:")
    print("=" * 60)
    print(f"总帧数: {len(results)}")
    print(f"识别文字总数: {total_texts}")
    print("=" * 60)
    
    # 保存结果
    if config.save_result:
        import json
        output_data = {
            'video_path': video_path,
            'frame_interval': frame_interval,
            'total_frames': len(results),
            'frames': [r.to_dict() for r in results]
        }
        with open('video_result.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info("结果已保存到: video_result.json")


def process_batch(ocr: OCRSystem, config: Config):
    """批量处理"""
    image_paths = config.batch_paths
    
    if not image_paths:
        logger.error("批量处理模式下，batch_paths 不能为空")
        logger.info("请在配置文件中设置 input.batch_paths")
        return
    
    logger.info(f"批量处理 {len(image_paths)} 张图片")
    
    # 执行OCR
    results = ocr.batch_process(image_paths)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("批量处理完成:")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"图片 {i} ({image_paths[i-1]}): {len(result.results)} 个文字")
    print("=" * 60)
    
    # 保存结果
    if config.save_result:
        import json
        output_data = {
            'total_images': len(results),
            'images': [r.to_dict() for r in results]
        }
        with open('batch_result.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info("结果已保存到: batch_result.json")


def process_stream(ocr: OCRSystem, config: Config):
    """处理数据流（摄像头/RTSP/RTMP）"""
    stream_source = config.stream_source
    max_frames = config.max_frames
    frame_interval = config.stream_frame_interval

    # 尝试将stream_source转换为整数（摄像头索引）
    try:
        stream_source = int(stream_source)
        logger.info(f"处理摄像头: {stream_source}")
    except (ValueError, TypeError):
        logger.info(f"处理数据流: {stream_source}")

    logger.info(f"帧间隔: {frame_interval}")
    logger.info(f"最大处理帧数: {max_frames if max_frames else '无限制'}")

    # 执行OCR
    frame_count = 0
    total_texts = 0

    print("\n" + "=" * 60)
    print("数据流处理中... (按 Ctrl+C 停止)")
    print(f"帧间隔: 每 {frame_interval} 帧处理一次")
    print("=" * 60)

    try:
        for result in ocr.process_stream(stream_source, max_frames, frame_interval):
            frame_count += 1
            text_count = len(result.results)
            total_texts += text_count

            # 打印当前帧结果
            print(f"\n帧 {frame_count}: 识别到 {text_count} 个文字")
            for r in result.results:
                print(f"  - {r.text} (置信度: {r.confidence:.2f})")

            # 保存结果（可选）
            if config.save_result and text_count > 0:
                import json
                output_file = f"stream_frame_{frame_count}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        logger.info("\n数据流处理被用户中断")

    # 打印摘要
    print("\n" + "=" * 60)
    print("数据流处理完成:")
    print("=" * 60)
    print(f"处理帧数: {frame_count}")
    print(f"识别文字总数: {total_texts}")
    print("=" * 60)


if __name__ == '__main__':
    main()
