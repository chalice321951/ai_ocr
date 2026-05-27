"""
配置管理模块
"""
import os
import yaml
from typing import Optional, Dict, Any


class Config:
    """系统配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        参数:
            config_path: 配置文件路径，None则使用默认配置
        """
        # 默认配置
        self.det_model_type = "DB"
        self.use_angle_cls = True
        self.rec_lang = "ch"
        self.use_gpu = True
        self.gpu_mem = 500
        self.confidence_threshold = 0.5
        self.output_format = "json"
        self.log_level = "INFO"
        self.log_file = "logs/ocr_system.log"
        self.batch_size = 8
        self.max_image_width = 1920
        self.max_image_height = 1080
        
        # 输入配置
        self.input_mode = "image"
        self.image_path = "test_image.jpg"
        self.video_path = "test_video.mp4"
        self.frame_interval = 30
        self.batch_paths = []
        self.stream_source = "0"  # 数据流源（摄像头索引或RTSP/RTMP URL）
        self.stream_frame_interval = 30  # 数据流帧间隔
        self.max_frames = 100  # 最大处理帧数
        
        # ROI 配置
        self.roi_enabled = False
        self.roi_hsv_lower = (0, 120, 100)
        self.roi_hsv_upper = (10, 255, 255)
        self.roi_min_area = 1000
        self.roi_dilate_kernel_size = 5
        self.roi_padding = 10
        self.roi_min_width_ratio = 0.3
        self.roi_max_height_ratio = 0.3

        # 站点匹配配置
        self.stations_file = "config/stations.json"

        # 输出配置
        self.save_result = True
        self.visualize = True
        self.save_roi_crop = True
        
        # 如果提供了配置文件路径，则加载配置
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """
        从YAML文件加载配置
        
        参数:
            config_path: 配置文件路径
            
        异常:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 加载检测配置
        if 'detection' in config_data:
            detection = config_data['detection']
            self.det_model_type = detection.get('model_type', self.det_model_type)
            self.use_angle_cls = detection.get('use_angle_cls', self.use_angle_cls)
        
        # 加载识别配置
        if 'recognition' in config_data:
            recognition = config_data['recognition']
            self.rec_lang = recognition.get('lang', self.rec_lang)
        
        # 加载系统配置
        if 'system' in config_data:
            system = config_data['system']
            self.use_gpu = system.get('use_gpu', self.use_gpu)
            self.gpu_mem = system.get('gpu_mem', self.gpu_mem)
            self.confidence_threshold = system.get('confidence_threshold', self.confidence_threshold)
            self.output_format = system.get('output_format', self.output_format)
        
        # 加载日志配置
        if 'logging' in config_data:
            logging = config_data['logging']
            self.log_level = logging.get('level', self.log_level)
            self.log_file = logging.get('log_file', self.log_file)
        
        # 加载性能配置
        if 'performance' in config_data:
            performance = config_data['performance']
            self.batch_size = performance.get('batch_size', self.batch_size)
            self.max_image_width = performance.get('max_image_width', self.max_image_width)
            self.max_image_height = performance.get('max_image_height', self.max_image_height)
        
        # 加载输入配置
        if 'input' in config_data:
            input_config = config_data['input']
            self.input_mode = input_config.get('mode', self.input_mode)
            self.image_path = input_config.get('image_path', self.image_path)
            self.video_path = input_config.get('video_path', self.video_path)
            self.frame_interval = input_config.get('frame_interval', self.frame_interval)
            self.batch_paths = input_config.get('batch_paths', self.batch_paths)
            self.stream_source = input_config.get('stream_source', self.stream_source)
            self.stream_frame_interval = input_config.get('stream_frame_interval', self.stream_frame_interval)
            self.max_frames = input_config.get('max_frames', self.max_frames)
        
        # 加载输出配置
        if 'output' in config_data:
            output_config = config_data['output']
            self.save_result = output_config.get('save_result', self.save_result)
            self.visualize = output_config.get('visualize', self.visualize)
            self.save_roi_crop = output_config.get('save_roi_crop', self.save_roi_crop)

        # 加载 ROI 配置
        if 'roi' in config_data:
            roi_config = config_data['roi']
            self.roi_enabled = roi_config.get('enabled', self.roi_enabled)
            if 'color_detect' in roi_config:
                cd = roi_config['color_detect']
                lower = cd.get('hsv_lower', list(self.roi_hsv_lower))
                upper = cd.get('hsv_upper', list(self.roi_hsv_upper))
                self.roi_hsv_lower = tuple(lower)
                self.roi_hsv_upper = tuple(upper)
                self.roi_min_area = cd.get('min_area', self.roi_min_area)
                self.roi_dilate_kernel_size = cd.get('dilate_kernel_size', self.roi_dilate_kernel_size)
                self.roi_padding = cd.get('padding', self.roi_padding)
                self.roi_min_width_ratio = cd.get('min_width_ratio', self.roi_min_width_ratio)
                self.roi_max_height_ratio = cd.get('max_height_ratio', self.roi_max_height_ratio)

        # 加载站点匹配配置
        if 'station_match' in config_data:
            sm_config = config_data['station_match']
            self.stations_file = sm_config.get('stations_file', self.stations_file)

        # 验证配置
        self._validate()
    
    def _validate(self) -> None:
        """
        验证配置的有效性
        
        异常:
            ValueError: 配置值无效
        """
        # 验证置信度阈值
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(f"置信度阈值必须在0-1之间: {self.confidence_threshold}")
        
        # 验证输出格式
        valid_formats = ['json', 'xml', 'csv']
        if self.output_format not in valid_formats:
            raise ValueError(f"输出格式必须是 {valid_formats} 之一: {self.output_format}")
        
        # 验证日志级别
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.log_level not in valid_levels:
            raise ValueError(f"日志级别必须是 {valid_levels} 之一: {self.log_level}")
        
        # 验证批次大小
        if self.batch_size <= 0:
            raise ValueError(f"批次大小必须大于0: {self.batch_size}")
    
    def save_to_file(self, config_path: str) -> None:
        """
        保存配置到YAML文件
        
        参数:
            config_path: 配置文件路径
        """
        config_data = {
            'detection': {
                'model_type': self.det_model_type,
                'use_angle_cls': self.use_angle_cls,
            },
            'recognition': {
                'lang': self.rec_lang,
            },
            'system': {
                'use_gpu': self.use_gpu,
                'gpu_mem': self.gpu_mem,
                'confidence_threshold': self.confidence_threshold,
                'output_format': self.output_format,
            },
            'logging': {
                'level': self.log_level,
                'log_file': self.log_file,
            },
            'performance': {
                'batch_size': self.batch_size,
                'max_image_width': self.max_image_width,
                'max_image_height': self.max_image_height,
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        返回:
            Dict[str, Any]: 配置字典
        """
        return {
            'det_model_type': self.det_model_type,
            'use_angle_cls': self.use_angle_cls,
            'rec_lang': self.rec_lang,
            'use_gpu': self.use_gpu,
            'gpu_mem': self.gpu_mem,
            'confidence_threshold': self.confidence_threshold,
            'output_format': self.output_format,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'batch_size': self.batch_size,
            'max_image_width': self.max_image_width,
            'max_image_height': self.max_image_height,
            'roi_enabled': self.roi_enabled,
            'roi_hsv_lower': self.roi_hsv_lower,
            'roi_hsv_upper': self.roi_hsv_upper,
            'save_roi_crop': self.save_roi_crop,
        }
