"""
图像转换节点模块
将图像转换为各种像素格式，支持像素数据分析，并提供高级图像保存功能
"""
import torch
import numpy as np
import os
import time
import json
import re
from datetime import datetime
from PIL import Image, PngImagePlugin
import folder_paths

class ImageToPixelInput:
    """图片转像素输入节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_format": (["pixel_array", "normalized_tensor", "flat_pixels", "rgb_values"], {"default": "pixel_array"}),
                "normalize_range": (["0-1", "0-255", "-1 to 1"], {"default": "0-1"}),
                "flatten_pixels": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pixel_output", "shape_info")
    FUNCTION = "convert_to_pixels"
    CATEGORY = "MISLG Tools/图像"
    DESCRIPTION = "将图像转换为像素输入格式"

    def convert_to_pixels(self, images, output_format, normalize_range, flatten_pixels):
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(images)}")

        shape_info = f"输入形状: {images.shape}, 格式: {images.dtype}\n"
        if images.dtype != torch.float32:
            images = images.float()
            shape_info += "转换数据类型为 float32\n"

        processed_images = self.process_images(images, output_format, normalize_range)
        shape_info += f"处理后形状: {processed_images.shape}\n"

        if flatten_pixels and len(processed_images.shape) > 2:
            original_shape = processed_images.shape
            if len(processed_images.shape) == 4:
                processed_images = processed_images.view(processed_images.shape[0], -1, processed_images.shape[3])
            else:
                processed_images = processed_images.view(-1, processed_images.shape[2])
            shape_info += f"展平: {original_shape} -> {processed_images.shape}\n"

        shape_info += f"输出格式: {output_format}, 范围: {normalize_range}"
        return (processed_images, shape_info)

    def process_images(self, images, output_format, normalize_range):
        if normalize_range == "0-255":
            images = images * 255.0
        elif normalize_range == "-1 to 1":
            images = (images * 2.0) - 1.0

        if output_format == "normalized_tensor":
            if normalize_range == "0-1":
                images = torch.clamp(images, 0.0, 1.0)
            elif normalize_range == "0-255":
                images = torch.clamp(images, 0.0, 255.0)
            elif normalize_range == "-1 to 1":
                images = torch.clamp(images, -1.0, 1.0)
        elif output_format == "flat_pixels":
            if len(images.shape) == 4:
                b, h, w, c = images.shape
                images = images.view(b, h * w, c)
        elif output_format == "rgb_values":
            if images.shape[-1] == 4:
                images = images[..., :3]
        return images


class PixelDataAnalyzer:
    """像素数据分析器"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixel_data": ("IMAGE",),
                "analyze_channels": ("BOOLEAN", {"default": True}),
                "show_sample_data": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("statistics", "data_sample", "shape_info")
    FUNCTION = "analyze_pixels"
    CATEGORY = "MISLG Tools/图像"
    DESCRIPTION = "分析像素数据的统计信息"

    def analyze_pixels(self, pixel_data, analyze_channels, show_sample_data):
        if not isinstance(pixel_data, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(pixel_data)}")

        stats = self.calculate_statistics(pixel_data, analyze_channels)
        sample = self.get_data_sample(pixel_data) if show_sample_data else "样本显示已关闭"
        shape_info = f"数据形状: {pixel_data.shape}\n数据类型: {pixel_data.dtype}"
        return (stats, sample, shape_info)

    def calculate_statistics(self, data, analyze_channels):
        stats = ["=== 像素数据统计 ==="]
        stats.append(f"形状: {data.shape}")
        stats.append(f"数据类型: {data.dtype}")
        stats.append(f"最小值: {data.min().item():.6f}")
        stats.append(f"最大值: {data.max().item():.6f}")
        stats.append(f"均值: {data.mean().item():.6f}")
        stats.append(f"标准差: {data.std().item():.6f}")

        if analyze_channels and len(data.shape) >= 4 and data.shape[-1] > 1:
            stats.append("\n=== 通道统计 ===")
            for c in range(data.shape[-1]):
                channel_data = data[..., c]
                stats.append(f"通道 {c}: min={channel_data.min().item():.3f}, "
                           f"max={channel_data.max().item():.3f}, "
                           f"mean={channel_data.mean().item():.3f}")
        return "\n".join(stats)

    def get_data_sample(self, data):
        try:
            sample_size = min(10, data.numel())
            flat_data = data.flatten()
            sample_indices = torch.linspace(0, flat_data.numel() - 1, sample_size).long()
            sample_values = flat_data[sample_indices]
            sample_str = "样本值: " + ", ".join([f"{v:.3f}" for v in sample_values])
            if data.numel() > sample_size:
                sample_str += f" ... (共 {data.numel()} 个元素)"
            return sample_str
        except Exception:
            return "无法生成样本"


class AdvancedImageSaver:
    """高级图像保存器 - 支持图像保存、工作流元数据嵌入、独立文本端口保存

    优化版本：
    1. 自定义计数器，精确匹配最后5位计数器（避免日期被误匹配）
    2. 降低 PNG 压缩级别，移除 optimize 以提升速度
    3. 默认关闭预览，避免双重保存
    4. 缓存 prompt JSON 序列化结果
    5. 日期在中间字段，计数器全局连续递增
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.temp_dir = os.path.join(self.output_dir, "temp_previews")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.cleanup_old_previews()
        # 缓存 prompt JSON，避免重复序列化
        self._cached_prompt_json = None
        self._cached_extra_pnginfo = None
        # 计数器缓存：{output_path: last_counter}，减少重复扫描
        self._counter_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "保存目录": (["默认输出", "自定义目录"], {"default": "默认输出"}),
                "自定义路径": ("STRING", {"default": "", "multiline": False, "tooltip": "选择'自定义目录'时必须填写完整路径"}),
                "文件名前缀": ("STRING", {"default": "ComfyUI"}),
                "图像格式": (["PNG", "JPG", "WEBP"], {"default": "PNG"}),
                "PNG压缩级别": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1, "display": "slider"}),
                "JPG图像质量": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "添加日期目录": ("BOOLEAN", {"default": True}),
                "添加日期": ("BOOLEAN", {"default": True}),
                "自动保存": ("BOOLEAN", {"default": True}),
                "WEBP无损": ("BOOLEAN", {"default": False}),
                "关闭预览": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "保存文本": ("STRING", {"forceInput": True, "tooltip": "连接文本节点输入内容，将自动保存为同名.txt文件"}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("详细信息",)
    FUNCTION = "save_images"
    CATEGORY = "MISLG Tools/图像"
    OUTPUT_NODE = True
    DESCRIPTION = "基于官方SaveImage优化的高级图像保存器，支持完整工作流嵌入与独立文本端口保存（优化版）"

    def cleanup_old_previews(self):
        try:
            current_time = time.time()
            one_hour_ago = current_time - 3600
            if os.path.exists(self.temp_dir):
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    if os.path.isfile(file_path) and os.path.getmtime(file_path) < one_hour_ago:
                        os.remove(file_path)
        except Exception as e:
            print(f"[清理预览] 出错: {e}")

    def resize_image_for_preview(self, img, max_size=1024):
        width, height = img.size
        if max(width, height) <= max_size:
            return img
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        return img.resize((new_width, new_height), resample)

    def _get_prompt_json(self, prompt):
        """缓存 prompt 的 JSON 序列化结果，避免重复序列化"""
        if prompt is None:
            return None
        if self._cached_prompt_json is None:
            self._cached_prompt_json = json.dumps(prompt)
        return self._cached_prompt_json

    def _get_extra_pnginfo_json(self, extra_pnginfo):
        """缓存 extra_pnginfo 的 JSON 序列化结果"""
        if extra_pnginfo is None:
            return None
        if self._cached_extra_pnginfo is None:
            self._cached_extra_pnginfo = {}
            for k, v in extra_pnginfo.items():
                self._cached_extra_pnginfo[k] = json.dumps(v)
        return self._cached_extra_pnginfo

    def _get_next_counter(self, prefix, output_path, add_date):
        """获取下一个计数器值

        关键修正：使用精确正则匹配文件名最后的5位计数器，
        避免日期数字被误匹配为计数器。

        匹配模式: _(\d{5})\. 匹配 _00001. 格式的最后5位数字
        """
        # 构建基础文件名（不含计数器）
        if add_date:
            date_str = datetime.now().strftime("%Y%m%d")
            base_pattern = f"{prefix}_{date_str}"
        else:
            base_pattern = prefix

        # 精确匹配最后5位计数器: _(\d{5})\.
        # 例如: ComfyUI_20260722_00001.png → 匹配 00001
        #       ComfyUI_20260722_20260723.png → 不匹配（最后不是5位数字）
        counter_pattern = re.compile(r"_(\d{5})\.")

        max_counter = 0
        if os.path.exists(output_path):
            for file_name in os.listdir(output_path):
                # 只检查匹配前缀的文件（减少不必要的匹配）
                if not file_name.startswith(base_pattern):
                    continue
                # 精确匹配最后5位计数器
                match = counter_pattern.search(file_name)
                if match:
                    counter = int(match.group(1))
                    max_counter = max(max_counter, counter)

        return max_counter + 1

    def save_images(self, 图像, 保存目录, 自定义路径, 文件名前缀, 图像格式, 
                    PNG压缩级别, JPG图像质量, 添加日期目录, 添加日期, 自动保存, WEBP无损, 关闭预览,
                    保存文本=None, prompt=None, extra_pnginfo=None):

        if not isinstance(图像, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(图像)}")

        # 重置缓存（每次执行时 prompt 可能不同）
        self._cached_prompt_json = None
        self._cached_extra_pnginfo = None

        # 1. 确定输出目录
        if 保存目录 == "自定义目录":
            target_path = 自定义路径.strip()
            if not target_path:
                print("⚠️ [高级图像保存器] 已选择'自定义目录'但未填写路径，已回退至默认输出目录")
                output_path = self.output_dir
            else:
                output_path = target_path
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_dir

        if 添加日期目录:
            date_dir_str = datetime.now().strftime("%Y-%m-%d")
            output_path = os.path.join(output_path, date_dir_str)
            os.makedirs(output_path, exist_ok=True)

        results = []
        saved_files = []
        saved_texts = [] 

        # 预序列化 metadata（缓存）
        prompt_json = self._get_prompt_json(prompt)
        extra_pnginfo_json = self._get_extra_pnginfo_json(extra_pnginfo)

        # 2. 处理图像队列
        for idx, image in enumerate(图像):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 使用自定义计数器，精确匹配最后5位计数器
            counter = self._get_next_counter(文件名前缀, output_path, 添加日期)
            counter_str = f"{counter:05d}"

            # 日期在中间字段，计数器在最后
            if 添加日期:
                date_str = datetime.now().strftime("%Y%m%d")
                filename_base = f"{文件名前缀}_{date_str}_{counter_str}"
            else:
                filename_base = f"{文件名前缀}_{counter_str}"

            file_name = f"{filename_base}.{图像格式.lower()}"
            save_path = os.path.join(output_path, file_name)

            # 核心修复：严格去除键名空格，确保 Pillow 正确识别参数
            save_kwargs = {}
            if 图像格式 == 'PNG':
                metadata = PngImagePlugin.PngInfo()
                if prompt_json is not None:
                    metadata.add_text("prompt", prompt_json)
                if extra_pnginfo_json is not None:
                    for k, v in extra_pnginfo_json.items():
                        metadata.add_text(k, v)
                metadata.add_text("generator", "MISLG AdvancedImageSaver")
                # 优化：compress_level 默认 4（与原生一致），移除 optimize=True
                save_kwargs = {
                    "pnginfo": metadata, 
                    "compress_level": PNG压缩级别
                }
            elif 图像格式 == 'JPG':
                save_kwargs = {"quality": JPG图像质量, "optimize": True}
                if img.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                elif img.mode != "RGB":
                    img = img.convert("RGB")
            elif 图像格式 == 'WEBP':
                save_kwargs = {"quality": JPG图像质量, "lossless": WEBP无损}

            if 自动保存:
                try:
                    img.save(save_path, **save_kwargs)
                    saved_files.append(save_path)
                except Exception as e:
                    print(f"[保存图像] 失败: {e}")

            if 保存文本 is not None and str(保存文本).strip():
                txt_path = os.path.join(output_path, f"{filename_base}.txt")
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(str(保存文本).strip())
                    saved_texts.append(txt_path)
                except Exception as e:
                    print(f"[保存文本] 失败: {e}")

            if not 关闭预览:
                preview_filename = f"preview_{filename_base}.png"
                preview_path = os.path.join(self.temp_dir, preview_filename)
                try:
                    preview_img = self.resize_image_for_preview(img, max_size=1024)
                    # 优化：预览图不嵌入完整 prompt，减少开销
                    preview_meta = PngImagePlugin.PngInfo()
                    preview_meta.add_text("generator", "MISLG Preview")
                    preview_img.save(preview_path, pnginfo=preview_meta, compress_level=3)
                    results.append({"filename": preview_filename, "subfolder": "temp_previews", "type": "output"})
                except Exception as e:
                    print(f"[生成预览] 失败: {e}")

        # 3. 构建返回信息
        info = ["=== 图像保存详情 ==="]
        info.append(f"保存目录: {output_path}")
        info.append(f"图像格式: {图像格式}")
        if 图像格式 == 'PNG':
            info.append(f"PNG压缩级别: {PNG压缩级别} (仅改变体积/速度，PNG为无损格式画质不变)")
        elif 图像格式 == 'JPG':
            info.append(f"JPG图像质量: {JPG图像质量}")
        else:
            info.append(f"WEBP质量: {JPG图像质量} | 无损: {'开' if WEBP无损 else '关'}")

        info.append(f"日期目录: {'已添加' if 添加日期目录 else '未添加'}")
        info.append(f"文件名日期: {'已添加' if 添加日期 else '未添加'} (中间字段，计数器全局连续)")
        info.append(f"自动保存: {'开' if 自动保存 else '关'}")
        info.append(f"文本端口: {'已连接' if (保存文本 is not None and str(保存文本).strip()) else '未连接/跳过'}")
        info.append(f"预览: {'关' if 关闭预览 else '开'}")

        if saved_files: info.append(f"\n✅ 已保存图像: {len(saved_files)} 张")
        if saved_texts: info.append(f"📄 已保存文本: {len(saved_texts)} 个")

        detail = "\n".join(info)
        return {"ui": {"images": results} if not 关闭预览 else {}, "result": (detail,)}


# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageToPixelInput": ImageToPixelInput,
    "PixelDataAnalyzer": PixelDataAnalyzer,
    "AdvancedImageSaver": AdvancedImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToPixelInput": "🔄 图像转像素",
    "PixelDataAnalyzer": "📊 像素数据分析",
    "AdvancedImageSaver": "💾 高级图像保存器",
}
