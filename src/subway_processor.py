"""
地铁站名识别后处理模块
"""
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .models import RecognitionResult
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class StationName:
    """地铁站名数据类"""
    chinese: str  # 中文站名
    english: str  # 英文站名
    confidence: float  # 置信度
    position: Tuple[int, int]  # 位置 (x, y)
    
    def __str__(self) -> str:
        return f"{self.chinese} {self.english} (置信度: {self.confidence:.2f})"


class SubwayProcessor:
    """地铁站名识别后处理器"""
    
    # 中文字符正则表达式
    CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]+')
    # 英文字符正则表达式
    ENGLISH_PATTERN = re.compile(r'[A-Za-z\s]+')
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        初始化地铁站名处理器
        
        参数:
            confidence_threshold: 置信度阈值，低于此值的结果将被过滤
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"初始化地铁站名处理器，置信度阈值: {confidence_threshold}")
    
    def parse_station_names(
        self, 
        results: List[RecognitionResult]
    ) -> List[StationName]:
        """
        解析地铁站名识别结果
        
        参数:
            results: OCR识别结果列表
            
        返回:
            List[StationName]: 解析后的站名列表
        """
        logger.info(f"开始解析 {len(results)} 个识别结果")
        
        # 1. 过滤低置信度结果
        filtered_results = self._filter_by_confidence(results)
        logger.info(f"过滤后剩余 {len(filtered_results)} 个结果")
        
        # 2. 按位置排序（从左到右，从上到下）
        sorted_results = self._sort_by_position(filtered_results)
        
        # 3. 解析中英文对应关系
        station_names = self._parse_bilingual_names(sorted_results)
        
        logger.info(f"成功解析 {len(station_names)} 个站名")
        for station in station_names:
            logger.debug(f"  {station}")
        
        return station_names
    
    def _filter_by_confidence(
        self, 
        results: List[RecognitionResult]
    ) -> List[RecognitionResult]:
        """
        根据置信度过滤结果
        
        参数:
            results: 识别结果列表
            
        返回:
            List[RecognitionResult]: 过滤后的结果列表
        """
        return [
            r for r in results 
            if r.confidence >= self.confidence_threshold
        ]
    
    def _sort_by_position(
        self, 
        results: List[RecognitionResult]
    ) -> List[RecognitionResult]:
        """
        按位置排序（从左到右，从上到下）
        
        参数:
            results: 识别结果列表
            
        返回:
            List[RecognitionResult]: 排序后的结果列表
        """
        # 按y坐标分组（允许一定误差）
        def get_row(result: RecognitionResult) -> int:
            # 将y坐标分组，每50像素为一行
            return result.bbox.y // 50
        
        # 先按行排序，再按x坐标排序
        return sorted(results, key=lambda r: (get_row(r), r.bbox.x))
    
    def _parse_bilingual_names(
        self, 
        results: List[RecognitionResult]
    ) -> List[StationName]:
        """
        解析中英文双语站名
        
        参数:
            results: 排序后的识别结果列表
            
        返回:
            List[StationName]: 站名列表
        """
        station_names = []
        i = 0
        
        while i < len(results):
            result = results[i]
            text = result.text.strip()
            
            # 尝试从单个文本中分离中英文
            chinese, english = self._split_bilingual_text(text)
            
            if chinese and english:
                # 单个文本包含中英文
                station = StationName(
                    chinese=chinese,
                    english=english,
                    confidence=result.confidence,
                    position=(result.bbox.x, result.bbox.y)
                )
                station_names.append(station)
                i += 1
            elif chinese:
                # 只有中文，尝试查找下一个英文
                english_result = self._find_next_english(results, i)
                if english_result:
                    station = StationName(
                        chinese=chinese,
                        english=english_result.text.strip(),
                        confidence=(result.confidence + english_result.confidence) / 2,
                        position=(result.bbox.x, result.bbox.y)
                    )
                    station_names.append(station)
                    i += 2  # 跳过下一个英文结果
                else:
                    # 没有找到对应的英文
                    station = StationName(
                        chinese=chinese,
                        english='',
                        confidence=result.confidence,
                        position=(result.bbox.x, result.bbox.y)
                    )
                    station_names.append(station)
                    i += 1
            elif english:
                # 只有英文，尝试查找前一个中文
                # 这种情况通常不会发生，因为我们是从左到右处理的
                station = StationName(
                    chinese='',
                    english=english,
                    confidence=result.confidence,
                    position=(result.bbox.x, result.bbox.y)
                )
                station_names.append(station)
                i += 1
            else:
                # 既不是中文也不是英文，跳过
                logger.warning(f"无法识别的文本: '{text}'")
                i += 1
        
        return station_names
    
    def _split_bilingual_text(self, text: str) -> Tuple[str, str]:
        """
        从单个文本中分离中英文
        
        参数:
            text: 文本字符串
            
        返回:
            Tuple[str, str]: (中文, 英文)
        """
        # 提取中文
        chinese_matches = self.CHINESE_PATTERN.findall(text)
        chinese = ''.join(chinese_matches) if chinese_matches else ''
        
        # 提取英文
        english_matches = self.ENGLISH_PATTERN.findall(text)
        english = ' '.join(english_matches).strip() if english_matches else ''
        
        return chinese, english
    
    def _find_next_english(
        self, 
        results: List[RecognitionResult], 
        current_index: int
    ) -> Optional[RecognitionResult]:
        """
        查找下一个英文文本
        
        参数:
            results: 识别结果列表
            current_index: 当前索引
            
        返回:
            Optional[RecognitionResult]: 下一个英文结果，如果没有则返回None
        """
        if current_index + 1 >= len(results):
            return None
        
        next_result = results[current_index + 1]
        next_text = next_result.text.strip()
        
        # 检查是否主要是英文
        if self.ENGLISH_PATTERN.search(next_text):
            # 检查位置是否接近（在同一行或相邻行）
            current_y = results[current_index].bbox.y
            next_y = next_result.bbox.y
            
            if abs(next_y - current_y) < 100:  # 允许100像素的垂直距离
                return next_result
        
        return None
    
    def filter_by_keywords(
        self,
        station_names: List[StationName],
        keywords: List[str]
    ) -> List[StationName]:
        """
        根据关键词过滤站名

        参数:
            station_names: 站名列表
            keywords: 关键词列表

        返回:
            List[StationName]: 过滤后的站名列表
        """
        filtered = []

        for station in station_names:
            for keyword in keywords:
                if keyword in station.chinese or keyword.lower() in station.english.lower():
                    filtered.append(station)
                    break

        logger.info(f"根据关键词过滤: {len(station_names)} -> {len(filtered)}")
        return filtered

    def filter_station_info(
        self,
        results: List[RecognitionResult]
    ) -> List[dict]:
        """
        从OCR结果中过滤出"当前站：XXX"或"下一站：XXX"格式的内容

        参数:
            results: OCR识别结果列表

        返回:
            List[dict]: 过滤后的结果列表，每个元素包含 type(当前站/下一站) 和 station_name(站名)
        """
        # 匹配模式：当前站：xxx 或 下一站：xxx
        pattern = re.compile(r'(当前站|下一站)[：:]\s*(.+)')

        filtered = []

        for result in results:
            text = result.text.strip()
            match = pattern.match(text)

            if match:
                station_type = match.group(1)  # "当前站" 或 "下一站"
                station_name = match.group(2).strip()  # 站名

                filtered.append({
                    'type': station_type,
                    'station_name': station_name,
                    'full_text': text,
                    'confidence': result.confidence,
                    'bbox': result.bbox
                })
                logger.info(f"匹配到: {station_type} -> {station_name} (置信度: {result.confidence:.2f})")

        logger.info(f"站点信息过滤: {len(results)} 个结果中找到 {len(filtered)} 个站点信息")
        return filtered
