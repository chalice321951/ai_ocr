"""
精度评估模块
"""
from typing import List, Tuple
from .models import AccuracyMetrics
from .logger import get_logger

logger = get_logger(__name__)


class AccuracyEvaluator:
    """精度评估器"""
    
    @staticmethod
    def character_accuracy(pred: str, gt: str) -> float:
        """
        计算字符级准确率（使用编辑距离）
        
        参数:
            pred: 预测文本
            gt: 真实文本（ground truth）
            
        返回:
            float: 字符级准确率 (0-1)
        """
        if not gt:
            return 1.0 if not pred else 0.0
        
        # 计算编辑距离
        edit_distance = AccuracyEvaluator._levenshtein_distance(pred, gt)
        
        # 字符级准确率 = 1 - (编辑距离 / 真实文本长度)
        accuracy = 1.0 - (edit_distance / len(gt))
        
        return max(0.0, accuracy)  # 确保不为负数
    
    @staticmethod
    def word_accuracy(pred: str, gt: str) -> float:
        """
        计算单词级准确率（完全匹配）
        
        参数:
            pred: 预测文本
            gt: 真实文本
            
        返回:
            float: 单词级准确率 (0或1)
        """
        # 去除首尾空格后比较
        return 1.0 if pred.strip() == gt.strip() else 0.0
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        计算两个字符串的编辑距离（Levenshtein距离）
        
        参数:
            s1: 字符串1
            s2: 字符串2
            
        返回:
            int: 编辑距离
        """
        if len(s1) < len(s2):
            return AccuracyEvaluator._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 插入、删除、替换的代价
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def evaluate(
        predictions: List[str], 
        ground_truths: List[str]
    ) -> AccuracyMetrics:
        """
        评估识别精度
        
        参数:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            
        返回:
            AccuracyMetrics: 精度指标
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"预测结果和真实标签数量不匹配: "
                f"{len(predictions)} vs {len(ground_truths)}"
            )
        
        if not predictions:
            logger.warning("评估数据为空")
            return AccuracyMetrics()
        
        total_samples = len(predictions)
        correct_samples = 0
        total_char_accuracy = 0.0
        total_word_accuracy = 0.0
        
        # 用于计算精确率和召回率
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(predictions, ground_truths):
            # 字符级准确率
            char_acc = AccuracyEvaluator.character_accuracy(pred, gt)
            total_char_accuracy += char_acc
            
            # 单词级准确率
            word_acc = AccuracyEvaluator.word_accuracy(pred, gt)
            total_word_accuracy += word_acc
            
            if word_acc == 1.0:
                correct_samples += 1
                true_positives += 1
            else:
                if pred:  # 有预测但不正确
                    false_positives += 1
                if gt:  # 有真实标签但未正确识别
                    false_negatives += 1
        
        # 计算各项指标
        accuracy = correct_samples / total_samples
        char_accuracy = total_char_accuracy / total_samples
        word_accuracy = total_word_accuracy / total_samples
        
        # 计算精确率
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        
        # 计算召回率
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        
        # 计算F1分数
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        metrics = AccuracyMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            char_accuracy=char_accuracy,
            word_accuracy=word_accuracy,
            total_samples=total_samples,
            correct_samples=correct_samples
        )
        
        logger.info(f"评估完成: {metrics}")
        
        return metrics
    
    @staticmethod
    def evaluate_by_language(
        predictions: List[str],
        ground_truths: List[str],
        language_labels: List[str]
    ) -> Tuple[AccuracyMetrics, AccuracyMetrics]:
        """
        按语言分别评估精度（中文和英文）
        
        参数:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            language_labels: 语言标签列表 ('zh' 或 'en')
            
        返回:
            Tuple[AccuracyMetrics, AccuracyMetrics]: (中文指标, 英文指标)
        """
        # 分离中文和英文数据
        zh_preds, zh_gts = [], []
        en_preds, en_gts = [], []
        
        for pred, gt, lang in zip(predictions, ground_truths, language_labels):
            if lang == 'zh':
                zh_preds.append(pred)
                zh_gts.append(gt)
            elif lang == 'en':
                en_preds.append(pred)
                en_gts.append(gt)
        
        # 分别评估
        zh_metrics = (
            AccuracyEvaluator.evaluate(zh_preds, zh_gts)
            if zh_preds
            else AccuracyMetrics()
        )
        
        en_metrics = (
            AccuracyEvaluator.evaluate(en_preds, en_gts)
            if en_preds
            else AccuracyMetrics()
        )
        
        logger.info(f"中文评估: {zh_metrics}")
        logger.info(f"英文评估: {en_metrics}")
        
        return zh_metrics, en_metrics
