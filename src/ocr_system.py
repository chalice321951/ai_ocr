"""
OCR系统主类 - 集成所有组件
"""
import os
import time
import cv2
import numpy as np
from typing import List, Iterator, Optional

from .config import Config
from .models import OCRResult, RecognitionResult, AccuracyMetrics
from .input_handler import ImageLoader, VideoProcessor, StreamProcessor
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .accuracy_evaluator import AccuracyEvaluator
from .logger import setup_logger, get_logger
from .color_detector import ColorDetector, ColorDetectorConfig
from .image_cropper import ImageCropper
from .output_manager import OutputManager
from .station_matcher import StationMatcher


class OCRSystem:
    """OCR文字识别系统"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化OCR系统
        
        参数:
            config: 配置对象，None则使用默认配置
        """
        # 加载配置
        self.config = config if config else Config()
        
        # 设置日志
        self.logger = setup_logger(
            name="OCRSystem",
            log_file=self.config.log_file,
            log_level=self.config.log_level
        )
        
        self.logger.info("初始化OCR系统")
        
        # 初始化输入处理器
        self.image_loader = ImageLoader()
        self.video_processor = VideoProcessor()
        self.stream_processor = StreamProcessor()
        
        # 初始化文本检测器
        self.text_detector = TextDetector(
            use_gpu=self.config.use_gpu,
            use_angle_cls=self.config.use_angle_cls
        )
        
        # 初始化文本识别器
        self.text_recognizer = TextRecognizer(
            lang=self.config.rec_lang,
            use_gpu=self.config.use_gpu,
            use_angle_cls=self.config.use_angle_cls
        )

        # 初始化 ROI 相关组件
        self.color_detector = None
        self.image_cropper = ImageCropper()
        self.output_manager = OutputManager()

        if self.config.roi_enabled:
            color_config = ColorDetectorConfig(
                hsv_lower=self.config.roi_hsv_lower,
                hsv_upper=self.config.roi_hsv_upper,
                min_area=self.config.roi_min_area,
                padding=self.config.roi_padding,
                max_height_ratio=self.config.roi_max_height_ratio,
            )
            self.color_detector = ColorDetector(color_config)
            self.logger.info("ROI 模式已启用")

        # 初始化站点匹配器
        self.station_matcher = StationMatcher(self.config.stations_file)

        # 变化检测状态（视频/流模式使用）
        self._last_text = ""

        self.logger.info("OCR系统初始化完成")
    
    def process_image(self, image_path: str) -> OCRResult:
        """
        处理单张图片

        参数:
            image_path: 图片文件路径

        返回:
            OCRResult: 包含检测框和识别文字的结果对象
        """
        self.logger.info(f"开始处理图片: {image_path}")
        start_time = time.time()

        try:
            # 1. 加载图片
            image = self.image_loader.load(image_path)

            if self.config.roi_enabled and self.color_detector is not None:
                return self._process_image_roi(image, image_path, start_time)

            # 2. 全图检测文字区域
            bboxes = self.text_detector.detect(image)

            # 3. 如果没有检测到文字，返回空结果
            if not bboxes:
                self.logger.info("未检测到文字区域")
                processing_time = time.time() - start_time
                return OCRResult(
                    results=[],
                    processing_time=processing_time,
                    image_path=image_path
                )

            # 4. 识别文字
            results = self.text_recognizer.recognize(image, bboxes)

            # 5. 根据置信度阈值过滤结果
            filtered_results = [
                r for r in results
                if r.confidence >= self.config.confidence_threshold
            ]

            processing_time = time.time() - start_time

            self.logger.info(
                f"图片处理完成: 检测到 {len(bboxes)} 个区域, "
                f"识别成功 {len(results)} 个, "
                f"过滤后 {len(filtered_results)} 个, "
                f"耗时 {processing_time:.2f}秒"
            )

            return OCRResult(
                results=filtered_results,
                processing_time=processing_time,
                image_path=image_path
            )

        except Exception as e:
            self.logger.error(f"处理图片失败: {str(e)}")
            raise

    def _process_image_roi(self, image, image_path, start_time):
        """单张图片 ROI 模式处理"""
        # 颜色检测定位 ROI
        roi = self.color_detector.detect(image)
        if roi is None:
            self.logger.info("颜色检测未找到目标区域")
            processing_time = time.time() - start_time
            return OCRResult(results=[], processing_time=processing_time, image_path=image_path)

        # 裁剪 ROI
        roi_image = self.image_cropper.crop(image, roi)

        # OCR 识别（直接对整个 ROI 做 OCR，不做预切割）
        roi_x, roi_y = roi[0], roi[1]
        results = self.text_recognizer.recognize_full(roi_image)
        for r in results:
            r.bbox.x += roi_x
            r.bbox.y += roi_y
        filtered_results = [r for r in results if r.confidence >= self.config.confidence_threshold]

        processing_time = time.time() - start_time

        # 保存输出
        if self.config.save_result:
            self.output_manager.start_session("roi_detect")

            if self.config.visualize:
                annotated = image.copy()
                x, y, w, h = roi
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for r in filtered_results:
                    bx, by, bw, bh = r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height
                    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                    cv2.putText(annotated, r.text, (bx, by - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                self.output_manager.save_annotated_image(annotated, "annotated.jpg")

            self.output_manager.save_ocr_result({
                "image_path": image_path,
                "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
                "results": [r.to_dict() for r in filtered_results],
                "processing_time": processing_time,
            })

            if self.config.save_roi_crop:
                self.output_manager.save_roi_crop(roi_image, "roi_0001.jpg")

        self.logger.info(
            f"图片 ROI 处理完成: ROI={roi}, "
            f"识别到 {len(filtered_results)} 条文字, "
            f"耗时 {processing_time:.2f}秒"
        )

        return OCRResult(results=filtered_results, processing_time=processing_time, image_path=image_path)
    
    def process_video(
        self,
        video_path: str,
        frame_interval: int = 1
    ) -> List[OCRResult]:
        """
        处理视频文件

        参数:
            video_path: 视频文件路径
            frame_interval: 帧间隔，每隔多少帧处理一次

        返回:
            List[OCRResult]: 每一帧的OCR结果列表
        """
        self.logger.info(f"开始处理视频: {video_path}, 帧间隔: {frame_interval}")
        start_time = time.time()

        results = []
        frame_count = 0

        # ROI 模式：初始化跟踪器
        use_roi = self.config.roi_enabled and self.color_detector is not None
        if use_roi:
            self._last_text = ""
            if self.config.save_result:
                self.output_manager.start_session("roi_detect")

        try:
            for frame in self.video_processor.extract_frames(video_path, frame_interval):
                frame_count += 1
                frame_start_time = time.time()

                if use_roi:
                    ocr_result = self._process_frame_roi(frame, frame_count, video_path)
                else:
                    ocr_result = self._process_frame_full(frame, f"{video_path}#frame{frame_count}")

                results.append(ocr_result)

            total_time = time.time() - start_time
            fps = frame_count / total_time if total_time > 0 else 0

            self.logger.info(
                f"视频处理完成: 处理 {frame_count} 帧, "
                f"总耗时 {total_time:.2f}秒, "
                f"平均 {fps:.2f} FPS"
            )

            return results

        except Exception as e:
            self.logger.error(f"处理视频失败: {str(e)}")
            raise
    
    def process_stream(
        self,
        stream_source,
        max_frames: Optional[int] = None,
        frame_interval: int = 1
    ) -> Iterator[OCRResult]:
        """
        处理数据流

        参数:
            stream_source: 数据流源
            max_frames: 最大处理帧数
            frame_interval: 帧间隔，每隔多少帧处理一次

        返回:
            Iterator[OCRResult]: OCR结果迭代器
        """
        self.logger.info(f"开始处理数据流: {stream_source}, 帧间隔: {frame_interval}")

        use_roi = self.config.roi_enabled and self.color_detector is not None
        if use_roi:
            self._last_text = ""
            if self.config.save_result:
                self.output_manager.start_session("roi_detect")

        try:
            frame_count = 0

            for frame in self.stream_processor.process_stream(stream_source, max_frames, frame_interval):
                frame_count += 1

                if use_roi:
                    ocr_result = self._process_frame_roi(frame, frame_count, "stream")
                else:
                    ocr_result = self._process_frame_full(frame, f"stream#frame{frame_count}")

                yield ocr_result

            self.logger.info(f"数据流处理完成: 共处理 {frame_count} 帧")

        except Exception as e:
            self.logger.error(f"处理数据流失败: {str(e)}")
            raise

    def _process_frame_full(self, frame, source_label):
        """全图模式处理单帧"""
        bboxes = self.text_detector.detect(frame)
        if bboxes:
            recognition_results = self.text_recognizer.recognize(frame, bboxes)
            filtered_results = [r for r in recognition_results if r.confidence >= self.config.confidence_threshold]
        else:
            filtered_results = []

        return OCRResult(results=filtered_results, processing_time=0, image_path=source_label)

    @staticmethod
    def _is_chinese(text):
        for ch in text:
            if '一' <= ch <= '鿿':
                return True
        return False

    def _get_station_text(self, results):
        """从识别结果中提取中文文本（按 x 坐标排序），并规范化站点前缀"""
        import re
        KNOWN_PREFIXES = ["当前站", "下一站", "本站", "上一站"]

        chinese = [r for r in results if self._is_chinese(r.text)]
        chinese.sort(key=lambda r: r.bbox.x)
        combined = "".join(r.text for r in chinese)

        # 规范化站点前缀
        for prefix in KNOWN_PREFIXES:
            pattern = re.escape(prefix) + r'[：:\s]*(.+)$'
            match = re.search(pattern, combined)
            if match:
                return f"{prefix}：{match.group(1).strip()}"
            if len(combined) >= 2 and combined in prefix:
                return prefix
            if len(combined) >= 2 and prefix.startswith(combined):
                return prefix

        return combined

    def _process_frame_roi(self, frame, frame_count, source_label):
        """ROI 模式处理单帧（每帧颜色检测，不用跟踪器）"""
        roi = self.color_detector.detect(frame)

        if roi is None:
            return OCRResult(results=[], processing_time=0, image_path=f"{source_label}#frame{frame_count}")

        roi_image = self.image_cropper.crop(frame, roi)

        roi_x, roi_y = roi[0], roi[1]
        results = self.text_recognizer.recognize_full(roi_image)
        for r in results:
            r.bbox.x += roi_x
            r.bbox.y += roi_y
        filtered_results = [r for r in results if r.confidence >= self.config.confidence_threshold]

        # 提取规范化文本
        station_text = self._get_station_text(filtered_results)
        # 用站点匹配器纠正
        station_text = self.station_matcher.match(station_text) or ""

        # 变化检测：文本没变或为空则不保存
        if self.config.save_result and station_text and station_text != self._last_text:
            self._last_text = station_text

            # 用时间戳+文本做子文件夹名
            from datetime import datetime
            ts = datetime.now().strftime("%H%M%S")
            safe_name = station_text.replace("：", "_").replace(":", "_").replace(" ", "_")
            sub_dir = os.path.join(self.output_manager.session_dir, f"{ts}_{safe_name}")
            os.makedirs(sub_dir, exist_ok=True)

            if self.config.visualize:
                annotated = frame.copy()
                x, y, w, h = roi
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                from PIL import Image, ImageDraw
                from .result_formatter import get_chinese_font
                pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                font = get_chinese_font(20)
                for r in filtered_results:
                    bx, by, bw, bh = r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height
                    draw.rectangle([bx, by, bx + bw, by + bh], outline=(0, 0, 255), width=2)
                    draw.text((bx, by - 22), r.text, fill=(0, 0, 255), font=font)
                annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(sub_dir, "annotated.jpg"), annotated)

            if self.config.save_roi_crop:
                cv2.imwrite(os.path.join(sub_dir, "roi_crop.jpg"), roi_image)

            import json
            with open(os.path.join(sub_dir, "ocr_result.json"), 'w', encoding='utf-8') as f:
                json.dump({
                    "frame": frame_count,
                    "source": source_label,
                    "text": station_text,
                    "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
                    "results": [r.to_dict() for r in filtered_results],
                }, f, ensure_ascii=False, indent=2)

        return OCRResult(results=filtered_results, processing_time=0, image_path=f"{source_label}#frame{frame_count}")
    
    def batch_process(
        self, 
        image_paths: List[str]
    ) -> List[OCRResult]:
        """
        批量处理多张图片
        
        参数:
            image_paths: 图片路径列表
            
        返回:
            List[OCRResult]: 所有图片的结果列表
        """
        self.logger.info(f"开始批量处理 {len(image_paths)} 张图片")
        start_time = time.time()
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"处理第 {i+1}/{len(image_paths)} 张图片")
                result = self.process_image(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"处理图片 {image_path} 失败: {str(e)}")
                # 添加空结果
                results.append(OCRResult(
                    results=[],
                    processing_time=0,
                    image_path=image_path
                ))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(image_paths) if image_paths else 0
        
        self.logger.info(
            f"批量处理完成: {len(image_paths)} 张图片, "
            f"总耗时 {total_time:.2f}秒, "
            f"平均 {avg_time:.2f}秒/张"
        )
        
        return results
    
    def evaluate_accuracy(
        self, 
        test_dataset: List[dict]
    ) -> AccuracyMetrics:
        """
        评估模型精度
        
        参数:
            test_dataset: 测试数据集，格式为 [{"image": "path/to/image.jpg", "ground_truth": "真实文字"}, ...]
            
        返回:
            AccuracyMetrics: 包含准确率、召回率、F1分数的指标对象
        """
        self.logger.info(f"开始评估模型精度，测试样本数: {len(test_dataset)}")
        
        predictions = []
        ground_truths = []
        
        for i, sample in enumerate(test_dataset):
            image_path = sample.get('image')
            ground_truth = sample.get('ground_truth', '')
            
            try:
                # 对图片进行OCR识别
                ocr_result = self.process_image(image_path)
                
                # 将所有识别结果拼接成一个字符串
                predicted_text = ' '.join(ocr_result.get_all_texts())
                
                predictions.append(predicted_text)
                ground_truths.append(ground_truth)
                
                self.logger.debug(
                    f"样本 {i+1}/{len(test_dataset)}: "
                    f"预测='{predicted_text}', 真实='{ground_truth}'"
                )
                
            except Exception as e:
                self.logger.error(f"处理测试样本 {i+1} 失败: {str(e)}")
                predictions.append('')
                ground_truths.append(ground_truth)
        
        # 使用AccuracyEvaluator计算指标
        metrics = AccuracyEvaluator.evaluate(predictions, ground_truths)
        
        self.logger.info(f"精度评估完成:\n{metrics}")
        
        return metrics
    
    def evaluate_accuracy_by_language(
        self,
        test_dataset: List[dict]
    ) -> tuple:
        """
        按语言分别评估精度（中文和英文）
        
        参数:
            test_dataset: 测试数据集，格式为 [{"image": "path", "ground_truth": "文字", "language": "zh"/"en"}, ...]
            
        返回:
            tuple: (中文指标, 英文指标)
        """
        self.logger.info(f"开始按语言评估模型精度，测试样本数: {len(test_dataset)}")
        
        predictions = []
        ground_truths = []
        language_labels = []
        
        for i, sample in enumerate(test_dataset):
            image_path = sample.get('image')
            ground_truth = sample.get('ground_truth', '')
            language = sample.get('language', 'zh')
            
            try:
                # 对图片进行OCR识别
                ocr_result = self.process_image(image_path)
                
                # 将所有识别结果拼接成一个字符串
                predicted_text = ' '.join(ocr_result.get_all_texts())
                
                predictions.append(predicted_text)
                ground_truths.append(ground_truth)
                language_labels.append(language)
                
            except Exception as e:
                self.logger.error(f"处理测试样本 {i+1} 失败: {str(e)}")
                predictions.append('')
                ground_truths.append(ground_truth)
                language_labels.append(language)
        
        # 按语言分别评估
        zh_metrics, en_metrics = AccuracyEvaluator.evaluate_by_language(
            predictions, ground_truths, language_labels
        )
        
        self.logger.info(f"中文精度评估:\n{zh_metrics}")
        self.logger.info(f"英文精度评估:\n{en_metrics}")
        
        return zh_metrics, en_metrics
