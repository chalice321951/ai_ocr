"""
OCR系统主类 - 集成所有组件
"""
import time
import numpy as np
from typing import List, Iterator, Optional

from .config import Config
from .models import OCRResult, RecognitionResult, AccuracyMetrics
from .input_handler import ImageLoader, VideoProcessor, StreamProcessor
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .accuracy_evaluator import AccuracyEvaluator
from .logger import setup_logger, get_logger


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
        
        self.logger.info("=" * 50)
        self.logger.info("初始化OCR系统")
        self.logger.info(f"配置: {self.config.to_dict()}")
        
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
        
        self.logger.info("OCR系统初始化完成")
        self.logger.info("=" * 50)
    
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
            
            # 2. 检测文字区域
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
        
        try:
            # 提取视频帧并处理
            for frame in self.video_processor.extract_frames(video_path, frame_interval):
                frame_count += 1
                
                # 检测文字区域
                bboxes = self.text_detector.detect(frame)
                
                # 如果检测到文字，进行识别
                if bboxes:
                    recognition_results = self.text_recognizer.recognize(frame, bboxes)
                    
                    # 过滤低置信度结果
                    filtered_results = [
                        r for r in recognition_results 
                        if r.confidence >= self.config.confidence_threshold
                    ]
                else:
                    filtered_results = []
                
                # 创建OCR结果
                ocr_result = OCRResult(
                    results=filtered_results,
                    processing_time=0,  # 单帧处理时间
                    image_path=f"{video_path}#frame{frame_count}"
                )
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

        try:
            frame_count = 0

            for frame in self.stream_processor.process_stream(stream_source, max_frames, frame_interval):
                frame_count += 1
                frame_start_time = time.time()

                # 检测文字区域
                bboxes = self.text_detector.detect(frame)

                # 如果检测到文字，进行识别
                if bboxes:
                    recognition_results = self.text_recognizer.recognize(frame, bboxes)

                    # 过滤低置信度结果
                    filtered_results = [
                        r for r in recognition_results
                        if r.confidence >= self.config.confidence_threshold
                    ]
                else:
                    filtered_results = []

                processing_time = time.time() - frame_start_time

                # 创建OCR结果
                ocr_result = OCRResult(
                    results=filtered_results,
                    processing_time=processing_time,
                    image_path=f"stream#frame{frame_count}"
                )

                yield ocr_result

            self.logger.info(f"数据流处理完成: 共处理 {frame_count} 帧")

        except Exception as e:
            self.logger.error(f"处理数据流失败: {str(e)}")
            raise
    
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
