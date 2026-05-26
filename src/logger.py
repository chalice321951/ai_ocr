"""
日志系统模块
"""
import logging
import os
from typing import Optional


def setup_logger(
    name: str = "OCRSystem",
    log_file: Optional[str] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    配置并返回日志记录器
    
    参数:
        name: 日志记录器名称
        log_file: 日志文件路径，None则只输出到控制台
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "OCRSystem") -> logging.Logger:
    """
    获取已配置的日志记录器
    
    参数:
        name: 日志记录器名称
        
    返回:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)
