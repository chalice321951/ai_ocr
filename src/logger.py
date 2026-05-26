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
    # 压制 PaddleOCR / PaddlePaddle 的冗余日志
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddle").setLevel(logging.ERROR)
    logging.getLogger("ppstructure").setLevel(logging.ERROR)

    logger = logging.getLogger(name)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
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
