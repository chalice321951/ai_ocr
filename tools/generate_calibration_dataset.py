"""
从RTMP视频流自动生成校准数据集
功能：采集视频帧 + 数据增强（旋转、光照、模糊等）
"""
import cv2
import numpy as np
import os
import time
from pathlib import Path
import argparse


class CalibrationDatasetGenerator:
    """校准数据集生成器"""

    def __init__(self, output_dir="dataset/calibration", target_count=200):
        """
        初始化生成器

        Args:
            output_dir: 输出目录
            target_count: 目标图片数量
        """
        self.output_dir = output_dir
        self.target_count = target_count
        os.makedirs(output_dir, exist_ok=True)

    def capture_frames(self, stream_url, frame_interval=30, max_frames=50):
        """
        从视频流采集帧

        Args:
            stream_url: RTMP/RTSP流地址或摄像头ID
            frame_interval: 采集间隔（每隔多少帧采集一次）
            max_frames: 最大采集帧数

        Returns:
            采集到的帧列表
        """
        print(f"\n{'='*60}")
        print(f"从视频流采集帧")
        print(f"{'='*60}")
        print(f"流地址: {stream_url}")
        print(f"采集间隔: 每{frame_interval}帧")
        print(f"目标数量: {max_frames}帧")

        # 尝试打开视频流
        try:
            # 如果是数字，转为整数（摄像头ID）
            try:
                stream_url = int(stream_url)
            except:
                pass

            cap = cv2.VideoCapture(stream_url)

            if not cap.isOpened():
                print(f"✗ 无法打开视频流: {stream_url}")
                return []

            print(f"✓ 视频流打开成功")

            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  分辨率: {width}x{height}")
            print(f"  帧率: {fps:.1f} FPS")

            frames = []
            frame_count = 0
            captured_count = 0

            print(f"\n开始采集... (按Ctrl+C停止)")

            while captured_count < max_frames:
                ret, frame = cap.read()

                if not ret:
                    print(f"\n✗ 读取帧失败或流结束")
                    break

                frame_count += 1

                # 按间隔采集
                if frame_count % frame_interval == 0:
                    frames.append(frame.copy())
                    captured_count += 1
                    print(f"  采集进度: {captured_count}/{max_frames}", end='\r')

            cap.release()

            print(f"\n✓ 采集完成: 共{len(frames)}帧")
            return frames

        except KeyboardInterrupt:
            print(f"\n⚠ 用户中断采集")
            cap.release()
            return frames

        except Exception as e:
            print(f"\n✗ 采集出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    def augment_brightness(self, image, factor):
        """调整亮度"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def augment_contrast(self, image, factor):
        """调整对比度"""
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    def augment_rotate(self, image, angle):
        """旋转图像"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))

    def augment_blur(self, image, kernel_size=5):
        """添加模糊"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def augment_noise(self, image, noise_level=10):
        """添加噪声"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    def generate_augmented_images(self, frames):
        """
        对采集的帧进行数据增强

        Args:
            frames: 原始帧列表

        Returns:
            增强后的图片列表
        """
        print(f"\n{'='*60}")
        print(f"数据增强")
        print(f"{'='*60}")
        print(f"原始帧数: {len(frames)}")

        augmented_images = []

        for idx, frame in enumerate(frames):
            # 1. 原图
            augmented_images.append(('original', frame))

            # 2. 亮度变化（模拟不同光照）
            for brightness_factor in [0.7, 1.3]:
                aug = self.augment_brightness(frame, brightness_factor)
                augmented_images.append((f'brightness_{brightness_factor}', aug))

            # 3. 对比度变化
            for contrast_factor in [0.8, 1.2]:
                aug = self.augment_contrast(frame, contrast_factor)
                augmented_images.append((f'contrast_{contrast_factor}', aug))

            # 4. 旋转（小角度）
            for angle in [-5, -2, 2, 5]:
                aug = self.augment_rotate(frame, angle)
                augmented_images.append((f'rotate_{angle}', aug))

            # 5. 轻微模糊
            aug = self.augment_blur(frame, 3)
            augmented_images.append(('blur', aug))

            # 6. 轻微噪声
            aug = self.augment_noise(frame, 5)
            augmented_images.append(('noise', aug))

            print(f"  处理进度: {idx+1}/{len(frames)}", end='\r')

        print(f"\n✓ 增强完成: {len(frames)} → {len(augmented_images)} 张图片")
        return augmented_images

    def save_images(self, images):
        """
        保存图片到磁盘

        Args:
            images: (名称, 图片)元组列表
        """
        print(f"\n{'='*60}")
        print(f"保存图片")
        print(f"{'='*60}")
        print(f"输出目录: {self.output_dir}")

        saved_count = 0

        for idx, (aug_type, image) in enumerate(images):
            if saved_count >= self.target_count:
                break

            filename = f"calib_{idx:04d}_{aug_type}.jpg"
            filepath = os.path.join(self.output_dir, filename)

            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            if (saved_count % 10 == 0) or (saved_count == len(images)):
                print(f"  保存进度: {saved_count}/{min(len(images), self.target_count)}", end='\r')

        print(f"\n✓ 保存完成: {saved_count} 张图片")
        return saved_count

    def generate_file_list(self):
        """生成图片路径列表文件"""
        print(f"\n{'='*60}")
        print(f"生成图片列表文件")
        print(f"{'='*60}")

        # 列表文件保存在输出目录的上级目录
        parent_dir = Path(self.output_dir).parent
        list_file = parent_dir / "calibration_list.txt"

        # 获取所有图片
        image_files = sorted(Path(self.output_dir).glob("*.jpg"))

        with open(list_file, 'w') as f:
            for img_path in image_files:
                # 使用绝对路径
                f.write(f"{img_path.absolute()}\n")

        print(f"✓ 列表文件: {list_file}")
        print(f"  图片数量: {len(image_files)}")

        return str(list_file)

    def run(self, stream_url, frame_interval=30, max_frames=50):
        """
        运行完整流程

        Args:
            stream_url: 视频流地址
            frame_interval: 采集间隔
            max_frames: 最大采集帧数
        """
        print("="*60)
        print("校准数据集自动生成工具")
        print("="*60)
        print(f"目标图片数量: {self.target_count}")
        print(f"输出目录: {self.output_dir}")

        # 1. 采集帧
        frames = self.capture_frames(stream_url, frame_interval, max_frames)

        if len(frames) == 0:
            print("\n✗ 未采集到任何帧，退出")
            return False

        # 2. 数据增强
        augmented_images = self.generate_augmented_images(frames)

        # 3. 保存图片
        saved_count = self.save_images(augmented_images)

        if saved_count == 0:
            print("\n✗ 未保存任何图片，退出")
            return False

        # 4. 生成列表文件
        list_file = self.generate_file_list()

        # 总结
        print("\n" + "="*60)
        print("生成完成")
        print("="*60)
        print(f"✓ 原始帧数: {len(frames)}")
        print(f"✓ 增强后图片: {len(augmented_images)}")
        print(f"✓ 保存图片: {saved_count}")
        print(f"✓ 列表文件: {list_file}")
        print("\n下一步: 运行模型转换")
        print("  python tool/convert_to_rknn.py")
        print("="*60)

        return True


def main():
    parser = argparse.ArgumentParser(description='从视频流生成校准数据集')
    parser.add_argument('stream_url', type=str,
                       help='RTMP/RTSP流地址，例如: rtmp://server/live/stream 或摄像头ID (0, 1...)')
    parser.add_argument('--output', type=str, default='dataset/calibration',
                       help='输出目录 (默认: dataset/calibration)')
    parser.add_argument('--target-count', type=int, default=200,
                       help='目标图片数量 (默认: 200)')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='采集间隔帧数 (默认: 30)')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='最大采集原始帧数 (默认: 50)')

    args = parser.parse_args()

    # 创建生成器
    generator = CalibrationDatasetGenerator(
        output_dir=args.output,
        target_count=args.target_count
    )

    # 运行
    success = generator.run(
        stream_url=args.stream_url,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys

    # 示例用法
    print("="*60)
    print("使用示例:")
    print("="*60)
    print("# 从RTMP流采集")
    print("python tool/generate_calibration_dataset.py rtmp://39.106.224.108:1937/live100/stream100")
    print("\n# 从摄像头采集")
    print("python tool/generate_calibration_dataset.py 0")
    print("\n# 自定义参数")
    print("python tool/generate_calibration_dataset.py rtmp://server/live/stream --target-count 300 --frame-interval 60")
    print("="*60)
    print()

    sys.exit(main())
