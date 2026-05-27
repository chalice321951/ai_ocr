"""
站点匹配模块 - 前缀识别 + 站名模糊匹配
"""
import json
import os
import re
from difflib import SequenceMatcher
from typing import Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)


class StationMatcher:
    """地铁站点匹配器"""

    # 前缀 → 锚点字
    PREFIX_ANCHORS = {
        "当前站": ["当", "前"],
        "下一站": ["下", "一"],
    }

    def __init__(self, stations_file: str = None):
        self.stations = []
        if stations_file and os.path.exists(stations_file):
            self._load_stations(stations_file)

    def _load_stations(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 支持平铺列表或按线路分组
        if isinstance(data, list):
            self.stations = data
        elif isinstance(data, dict):
            for station_list in data.values():
                if isinstance(station_list, list):
                    self.stations.extend(station_list)
        # 去重
        self.stations = list(dict.fromkeys(self.stations))
        logger.info(f"加载站点列表: {len(self.stations)} 个站名")

    def match(self, ocr_text: str) -> Optional[str]:
        """
        对 OCR 文本做前缀识别 + 站名匹配。

        参数:
            ocr_text: OCR 拼接后的中文文本，如 "当前站：轨道大厦"

        返回:
            匹配成功返回规范化文本（如 "当前站：轨道大厦"），失败返回 None
        """
        if not ocr_text or not self.stations:
            return None

        # 1. 提取前缀
        prefix = self._detect_prefix(ocr_text)
        if prefix is None:
            return None

        # 2. 提取站名部分（去掉前缀和标点）
        station_name = self._extract_station_name(ocr_text)
        if not station_name:
            return None

        # 3. 匹配站名
        matched = self._match_station(station_name)
        if matched is None:
            return None

        return f"{prefix}：{matched}"

    def _detect_prefix(self, text: str) -> Optional[str]:
        """用锚点字检测前缀"""
        for prefix, anchors in self.PREFIX_ANCHORS.items():
            for anchor in anchors:
                if anchor in text:
                    return prefix
        return None

    def _extract_station_name(self, text: str) -> str:
        """从 OCR 文本中提取站名部分（去掉前缀和标点）"""
        # 去掉已知前缀
        for prefix in self.PREFIX_ANCHORS:
            text = text.replace(prefix, "")
        # 去掉冒号、空格等标点
        text = re.sub(r"[：:\s\-]", "", text)
        return text.strip()

    def _match_station(self, name: str) -> Optional[str]:
        """在站点列表中匹配站名"""
        if not name:
            return None

        # 1. 精确匹配
        if name in self.stations:
            return name

        # 2. OCR 站名是列表中某站名的前缀（"轨道" → "轨道大厦"）
        for station in self.stations:
            if station.startswith(name) and len(name) >= 2:
                return station

        # 3. 列表中某站名是 OCR 站名的前缀（"轨道大厦站" → "轨道大厦"）
        for station in self.stations:
            if name.startswith(station):
                return station

        # 4. 模糊匹配（字符重叠率 > 0.6）
        best_match = None
        best_ratio = 0.0
        for station in self.stations:
            ratio = SequenceMatcher(None, name, station).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = station

        if best_ratio > 0.6:
            return best_match

        return None
