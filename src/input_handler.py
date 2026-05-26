"""
输入处理模块 - 处理图片、视频和数据流输入
"""
import os
import cv2
import numpy as np
from typing import Iterator, Optional
from PIL import Image

from .exceptions import FileLoadError, UnsupportedFormatError
from .logger import get_logger

logger = get_logger(__name__)


class ImageLoader:
    """图片加载器"""
    
    # 支持的图片格式
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def load(self, image_path: str) -> np.ndarray:
        """
        加载图片文件
        
        参数:
            image_path: 图片路径
            
        返回:
            np.ndarray: 图像数组 (H, W, C)，BGR格式
            
        异常:
            FileLoadError: 文件不存在或无法读取
            UnsupportedFormatError: 不支持的文件格式
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            raise FileLoadError(image_path, "文件不存在")
        
        # 检查文件格式
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            logger.error(f"不支持的图片格式: {file_ext}")
            raise UnsupportedFormatError(file_ext, self.SUPPORTED_FORMATS)
        
        try:
            # 使用OpenCV加载图片
            image = cv2.imread(image_path)
            
            if image is None:
                # 如果OpenCV失败，尝试使用Pillow
                logger.warning(f"OpenCV加载失败，尝试使用Pillow: {image_path}")
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            if image is None:
                raise FileLoadError(image_path, "无法读取图片内容")
            
            logger.info(f"成功加载图片: {image_path}, 尺寸: {image.shape}")
            return image
            
        except Exception as e:
            if isinstance(e, (FileLoadError, UnsupportedFormatError)):
                raise
            logger.error(f"加载图片时发生错误: {image_path}, 错误: {str(e)}")
            raise FileLoadError(image_path, str(e))


class VideoProcessor:
    """视频处理器"""
    
    # 支持的视频格式
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def extract_frames(
        self, 
        video_path: str, 
        frame_interval: int = 1
    ) -> Iterator[np.ndarray]:
        """
        从视频中提取帧
        
        参数:
            video_path: 视频路径
            frame_interval: 帧间隔，每隔多少帧提取一次（1表示每帧都提取）
            
        返回:
            Iterator[np.ndarray]: 帧图像迭代器
            
        异常:
            FileLoadError: 文件不存在或无法读取
            UnsupportedFormatError: 不支持的视频格式
        """
        # 检查文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            raise FileLoadError(video_path, "文件不存在")
        
        # 检查文件格式
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            logger.error(f"不支持的视频格式: {file_ext}")
            raise UnsupportedFormatError(file_ext, self.SUPPORTED_FORMATS)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            raise FileLoadError(video_path, "无法打开视频文件")
        
        try:
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"开始处理视频: {video_path}, 总帧数: {total_frames}, FPS: {fps}")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 根据帧间隔提取帧
                if frame_count % frame_interval == 0:
                    extracted_count += 1
                    logger.debug(f"提取第 {frame_count} 帧 (第 {extracted_count} 个提取帧)")
                    yield frame
                
                frame_count += 1
            
            logger.info(f"视频处理完成: 总帧数 {frame_count}, 提取帧数 {extracted_count}")
            
        finally:
            cap.release()


class StreamProcessor:
    """数据流处理器"""

    def process_stream(
        self,
        stream_source,
        max_frames: Optional[int] = None,
        frame_interval: int = 1
    ) -> Iterator[np.ndarray]:
        """
        处理数据流

        参数:
            stream_source: 数据流源（可以是摄像头索引、RTSP URL等）
            max_frames: 最大处理帧数，None表示无限制
            frame_interval: 帧间隔，每隔多少帧处理一次（1表示每帧都处理）

        返回:
            Iterator[np.ndarray]: 图像帧迭代器

        异常:
            FileLoadError: 无法打开数据流
        """
        # 打开数据流
        if isinstance(stream_source, int):
            logger.info(f"打开摄像头: {stream_source}")
        else:
            logger.info(f"打开数据流: {stream_source}")

        cap = cv2.VideoCapture(stream_source)

        if not cap.isOpened():
            logger.error(f"无法打开数据流: {stream_source}")
            raise FileLoadError(str(stream_source), "无法打开数据流")

        try:
            frame_count = 0
            processed_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    logger.warning("数据流读取失败或结束")
                    break

                frame_count += 1

                # 根据帧间隔处理帧
                if frame_count % frame_interval == 0:
                    processed_count += 1
                    logger.debug(f"处理数据流第 {frame_count} 帧 (第 {processed_count} 个处理帧)")
                    yield frame

                # 检查是否达到最大处理帧数（注意是处理帧数，不是总帧数）
                if max_frames and processed_count >= max_frames:
                    logger.info(f"达到最大处理帧数限制: {max_frames}")
                    break

            logger.info(f"数据流处理完成，总帧数 {frame_count}，处理帧数 {processed_count}")

        finally:
            cap.release()
