"""
自定义异常类模块
"""


class OCRException(Exception):
    """OCR系统基础异常类"""
    
    def __init__(self, message: str):
        """
        初始化异常
        
        参数:
            message: 错误消息
        """
        self.message = message
        super().__init__(self.message)


class FileLoadError(OCRException):
    """文件加载错误"""
    
    def __init__(self, file_path: str, reason: str = ""):
        """
        初始化文件加载错误
        
        参数:
            file_path: 文件路径
            reason: 错误原因
        """
        message = f"无法加载文件: {file_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)
        self.file_path = file_path
        self.reason = reason


class ModelLoadError(OCRException):
    """模型加载错误"""
    
    def __init__(self, model_name: str, reason: str = ""):
        """
        初始化模型加载错误
        
        参数:
            model_name: 模型名称
            reason: 错误原因
        """
        message = f"无法加载模型: {model_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message)
        self.model_name = model_name
        self.reason = reason


class InsufficientMemoryError(OCRException):
    """内存不足错误"""
    
    def __init__(self, required_memory: str = "", available_memory: str = ""):
        """
        初始化内存不足错误
        
        参数:
            required_memory: 所需内存
            available_memory: 可用内存
        """
        message = "内存不足，无法完成操作"
        if required_memory and available_memory:
            message += f" (需要: {required_memory}, 可用: {available_memory})"
        super().__init__(message)
        self.required_memory = required_memory
        self.available_memory = available_memory


class UnsupportedFormatError(OCRException):
    """不支持的格式错误"""
    
    def __init__(self, file_format: str, supported_formats: list = None):
        """
        初始化不支持的格式错误
        
        参数:
            file_format: 文件格式
            supported_formats: 支持的格式列表
        """
        message = f"不支持的文件格式: {file_format}"
        if supported_formats:
            message += f" (支持的格式: {', '.join(supported_formats)})"
        super().__init__(message)
        self.file_format = file_format
        self.supported_formats = supported_formats
