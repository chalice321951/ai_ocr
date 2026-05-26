"""
示例：评估模型精度
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.ocr_system import OCRSystem


def main():
    """评估模型精度的示例"""
    
    # 1. 创建配置
    config = Config()
    config.use_gpu = True
    config.confidence_threshold = 0.5
    
    # 2. 初始化OCR系统
    print("初始化OCR系统...")
    ocr = OCRSystem(config)
    
    # 3. 准备测试数据集
    # 格式: [{"image": "图片路径", "ground_truth": "真实文字"}, ...]
    test_dataset = [
        {"image": "test1.jpg", "ground_truth": "大屯河街"},
        {"image": "test2.jpg", "ground_truth": "Dayanhe Street"},
        {"image": "test3.jpg", "ground_truth": "金华南街 Jinhuanan Street"},
        # 添加更多测试样本...
    ]
    
    print(f"\n测试数据集: {len(test_dataset)} 个样本")
    
    # 检查测试图片是否存在
    missing_files = [s['image'] for s in test_dataset if not os.path.exists(s['image'])]
    if missing_files:
        print(f"\n警告：以下测试图片不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请准备测试图片，或修改test_dataset变量")
        return
    
    # 4. 评估精度
    print("\n开始评估...")
    metrics = ocr.evaluate_accuracy(test_dataset)
    
    # 5. 打印结果
    print("\n" + "=" * 50)
    print("精度评估结果:")
    print("=" * 50)
    print(metrics)
    print("=" * 50)
    
    # 6. 按语言分别评估（如果有语言标签）
    print("\n按语言分别评估...")
    
    # 为测试数据添加语言标签
    test_dataset_with_lang = [
        {"image": "test1.jpg", "ground_truth": "大屯河街", "language": "zh"},
        {"image": "test2.jpg", "ground_truth": "Dayanhe Street", "language": "en"},
        {"image": "test3.jpg", "ground_truth": "金华南街 Jinhuanan Street", "language": "zh"},
    ]
    
    zh_metrics, en_metrics = ocr.evaluate_accuracy_by_language(test_dataset_with_lang)
    
    print("\n" + "=" * 50)
    print("中文识别精度:")
    print("=" * 50)
    print(zh_metrics)
    print()
    print("=" * 50)
    print("英文识别精度:")
    print("=" * 50)
    print(en_metrics)
    print("=" * 50)
    
    # 7. 保存评估报告
    import json
    report = {
        'overall': metrics.to_dict(),
        'chinese': zh_metrics.to_dict(),
        'english': en_metrics.to_dict()
    }
    
    with open('accuracy_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n评估报告已保存到: accuracy_report.json")


if __name__ == '__main__':
    main()
