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
from src.logger import setup_logger
from src.output_manager import OutputManager
from src.station_matcher import StationMatcher

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


def is_chinese(text):
    """判断文本是否包含中文字符"""
    for ch in text:
        if '一' <= ch <= '鿿':
            return True
    return False


def process_image(ocr: OCRSystem, config: Config):
    """处理单张图片"""
    image_path = config.image_path

    logger.info(f"处理: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在: {image_path}")
        return

    result = ocr.process_image(image_path)

    # 过滤掉纯英文结果，只保留含中文的文字，按 x 坐标从左到右排序
    chinese_results = [r for r in result.results if is_chinese(r.text)]
    chinese_results.sort(key=lambda r: r.bbox.x)

    # 按位置顺序拼接中文文字
    combined_text = "".join(r.text for r in chinese_results)

    # 站点匹配
    matcher = StationMatcher(config.stations_file)
    station_text = matcher.match(combined_text) or combined_text

    print(f"\n识别结果 ({result.processing_time:.2f}秒):")
    print(f"  {station_text}")

    # 保存结果到 res 目录
    if config.save_result:
        from datetime import datetime
        output_mgr = OutputManager()
        detect_type = "roi_detect" if config.roi_enabled else "full_detect"
        session_dir = output_mgr.start_session(detect_type)

        # 创建子文件夹：时间戳_检测结果
        ts = datetime.now().strftime("%H%M%S")
        safe_name = station_text.replace("：", "_").replace(":", "_").replace(" ", "_")
        sub_dir = os.path.join(session_dir, f"{ts}_{safe_name}")
        os.makedirs(sub_dir, exist_ok=True)

        import json
        json_data = {
            "text": station_text,
            "results": [r.to_dict() for r in result.results],
            "processing_time": result.processing_time,
        }
        json_path = os.path.join(sub_dir, "ocr_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {json_path}")

        if config.visualize:
            try:
                image = cv2.imread(image_path)
                vis_image = ResultFormatter.visualize(image, chinese_results)
                vis_path = os.path.join(sub_dir, "annotated.jpg")
                cv2.imwrite(vis_path, vis_image)
                logger.info(f"可视化已保存: {vis_path}")
            except Exception as e:
                logger.warning(f"可视化失败: {e}")


def process_video(ocr: OCRSystem, config: Config):
    """处理视频"""
    video_path = config.video_path
    frame_interval = config.frame_interval

    logger.info(f"处理视频: {video_path}, 帧间隔: {frame_interval}")

    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return

    ocr.process_video(video_path, frame_interval)
    print("\n视频处理完成")


def process_batch(ocr: OCRSystem, config: Config):
    """批量处理"""
    image_paths = config.batch_paths

    if not image_paths:
        logger.error("批量处理模式下，batch_paths 不能为空")
        return

    logger.info(f"批量处理 {len(image_paths)} 张图片")
    ocr.batch_process(image_paths)
    print("\n批量处理完成")


def process_stream(ocr: OCRSystem, config: Config):
    """处理数据流（摄像头/RTSP/RTMP）"""
    stream_source = config.stream_source
    max_frames = config.max_frames
    frame_interval = config.stream_frame_interval

    try:
        stream_source = int(stream_source)
        logger.info(f"处理摄像头: {stream_source}")
    except (ValueError, TypeError):
        logger.info(f"处理数据流: {stream_source}")

    logger.info(f"帧间隔: {frame_interval}, 最大帧数: {max_frames or '无限制'}")
    print("\n数据流处理中... (按 Ctrl+C 停止)\n")

    matcher = StationMatcher(config.stations_file)
    last_text = ""
    frame_count = 0

    try:
        for result in ocr.process_stream(stream_source, max_frames, frame_interval):
            frame_count += 1

            # 过滤英文，按 x 排序，拼接中文
            chinese = [r for r in result.results if is_chinese(r.text)]
            chinese.sort(key=lambda r: r.bbox.x)
            combined = "".join(r.text for r in chinese)

            # 站点匹配
            matched = matcher.match(combined)
            if matched and matched != last_text:
                last_text = matched
                logger.info(f"检测到变化: {matched} (帧 {frame_count})")

    except KeyboardInterrupt:
        logger.info("\n数据流处理被用户中断")

    print(f"\n处理完成: 共 {frame_count} 帧")


if __name__ == '__main__':
    main()
