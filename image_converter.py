"""
图像转换节点模块
将图像转换为各种像素格式，支持像素数据分析，并提供高级图像保存功能
"""

import torch
import numpy as np
import os
import time
import json
import tempfile
import shutil
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
                "output_format": (["pixel_array", "normalized_tensor", "flat_pixels", "rgb_values"], 
                                {"default": "pixel_array"}),
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
        shape_info = f"输入形状: {images.shape}, 格式: {images.dtype}\n"
        
        if images.dtype != torch.float32:
            images = images.float()
            shape_info += f"转换数据类型为 float32\n"
        
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
            if normalize_range != "0-1":
                images = torch.clamp(images, 0.0, 1.0)
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
        stats = self.calculate_statistics(pixel_data, analyze_channels)
        sample = self.get_data_sample(pixel_data) if show_sample_data else "样本显示已关闭"
        shape_info = f"数据形状: {pixel_data.shape}\n数据类型: {pixel_data.dtype}"
        
        return (stats, sample, shape_info)

    def calculate_statistics(self, data, analyze_channels):
        stats = []
        stats.append("=== 像素数据统计 ===")
        stats.append(f"形状: {data.shape}")
        stats.append(f"数据类型: {data.dtype}")
        stats.append(f"最小值: {data.min().item():.6f}")
        stats.append(f"最大值: {data.max().item():.6f}")
        stats.append(f"均值: {data.mean().item():.6f}")
        stats.append(f"标准差: {data.std().item():.6f}")
        
        if analyze_channels and len(data.shape) > 1 and data.shape[-1] > 1:
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
            sample_indices = torch.linspace(0, flat_data.numel()-1, sample_size).long()
            sample_values = flat_data[sample_indices]
            
            sample_str = "样本值: " + ", ".join([f"{v:.3f}" for v in sample_values])
            if data.numel() > sample_size:
                sample_str += f" ... (共 {data.numel()} 个元素)"
                
            return sample_str
        except:
            return "无法生成样本"

class AdvancedImageSaver:
    """高级图像保存器 - 基于官方SaveImage优化，支持工作流嵌入"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        # 创建临时预览目录
        self.temp_dir = os.path.join(self.output_dir, "temp_previews")
        os.makedirs(self.temp_dir, exist_ok=True)
        # 清理旧的预览文件
        self.cleanup_old_previews()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "保存目录": (["默认输出", "自定义目录"], {"default": "默认输出"}),
                "文件名前缀": ("STRING", {"default": "ComfyUI"}),
                "图像格式": (["PNG", "JPG", "WEBP"], {"default": "PNG"}),
                "PNG压缩级别": ("INT", {"default": 6, "min": 0, "max": 9, "step": 1, "display": "slider"}),
                "图像质量": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "添加日期目录": ("BOOLEAN", {"default": True}),
                "添加日期": ("BOOLEAN", {"default": True}),
                "自动保存": ("BOOLEAN", {"default": True}),
                "WEBP无损": ("BOOLEAN", {"default": False}),
                "关闭预览": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "自定义路径": ("STRING", {"default": "", "multiline": False}),
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
    DESCRIPTION = "基于官方SaveImage优化的高级图像保存器，支持完整工作流嵌入"
    
    def cleanup_old_previews(self):
        """清理旧的预览文件，保留最近一小时的预览"""
        try:
            current_time = time.time()
            one_hour_ago = current_time - 1800  # 1小时前
            
            if os.path.exists(self.temp_dir):
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    if os.path.isfile(file_path):
                        # 检查文件创建时间
                        file_time = os.path.getctime(file_path)
                        if file_time < one_hour_ago:
                            os.remove(file_path)
                            print(f"已清理旧预览文件: {filename}")
        except Exception as e:
            print(f"清理预览文件时出错: {e}")
    
    def resize_image_for_preview(self, img, max_size=1024):
        """调整图像大小，使长边不超过max_size，保持宽高比"""
        width, height = img.size
        
        # 如果图像尺寸小于等于max_size，无需调整
        if max(width, height) <= max_size:
            return img
        
        # 计算新的尺寸
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        # 使用LANCZOS重采样进行高质量缩放
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def get_next_counter(self, prefix, output_path, add_date):
        """获取下一个可用的计数器值，支持两种模式：
        1. 添加日期时：每天从00001开始计数，格式：前缀_日期_XXXXX.后缀
        2. 不添加日期时：从现有文件中累加计数，格式：前缀_XXXXX.后缀
        """
        if add_date:
            # 添加日期时，每天从00001开始
            date_str = datetime.now().strftime("%Y%m%d")
            filename_prefix = f"{prefix}_{date_str}"
            base_name = filename_prefix
        else:
            # 不添加日期时，直接使用前缀
            filename_prefix = prefix
            base_name = prefix
        
        # 扫描目录中的现有文件，找到最大的计数器值
        max_counter = 0
        
        if os.path.exists(output_path):
            for file_name in os.listdir(output_path):
                # 去掉扩展名
                file_name_without_ext = os.path.splitext(file_name)[0]
                
                # 检查是否以指定前缀开头
                if file_name_without_ext.startswith(base_name):
                    # 提取计数器部分
                    if add_date:
                        # 格式：前缀_日期_XXXXX
                        # 需要检查是否包含日期部分
                        expected_prefix = f"{prefix}_{date_str}_"
                        if file_name_without_ext.startswith(expected_prefix):
                            # 提取计数器部分
                            counter_part = file_name_without_ext[len(expected_prefix):]
                            if counter_part.isdigit() and len(counter_part) == 5:
                                counter_value = int(counter_part)
                                if counter_value > max_counter:
                                    max_counter = counter_value
                    else:
                        # 格式：前缀_XXXXX
                        expected_prefix = f"{prefix}_"
                        if file_name_without_ext.startswith(expected_prefix):
                            # 提取计数器部分
                            counter_part = file_name_without_ext[len(expected_prefix):]
                            if counter_part.isdigit() and len(counter_part) == 5:
                                counter_value = int(counter_part)
                                if counter_value > max_counter:
                                    max_counter = counter_value
        
        # 下一个计数器值是最大值加1
        next_counter = max_counter + 1
        
        return base_name, next_counter
    
    def save_images(self, 图像, 保存目录, 文件名前缀, 图像格式, PNG压缩级别, 图像质量, 添加日期目录, 添加日期, 自动保存, WEBP无损, 关闭预览,
                   自定义路径="", prompt=None, extra_pnginfo=None):
        """保存图像 - 基于官方实现优化"""
        
        # 确定输出目录
        if 保存目录 == "自定义目录" and 自定义路径.strip():
            output_path = 自定义路径.strip()
            # 创建自定义目录
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_dir
        
        # 添加日期目录
        if 添加日期目录:
            date_dir_str = datetime.now().strftime("%Y-%m-%d")
            output_path = os.path.join(output_path, date_dir_str)
            os.makedirs(output_path, exist_ok=True)
        
        results = list()
        saved_files = []
        
        # 处理每张图像
        for idx, image in enumerate(图像):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # 获取下一个计数器值
            base_name, counter = self.get_next_counter(文件名前缀, output_path, 添加日期)
            
            # 构建文件名基础部分（不含扩展名）
            counter_str = f"{counter:05d}"
            
            if 添加日期:
                # 添加日期时：前缀_日期_XXXXX
                # base_name 已经是 前缀_日期
                filename_base = f"{base_name}_{counter_str}"
            else:
                # 不添加日期时：前缀_XXXXX
                filename_base = f"{base_name}_{counter_str}"
            
            # 生成元数据
            metadata = PngImagePlugin.PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            
            # 添加生成器信息
            metadata.add_text("generator", "MISLG AdvancedImageSaver")
            
            # 根据格式保存
            file = f"{filename_base}.{图像格式.lower()}"
            save_path = os.path.join(output_path, file)
            
            save_kwargs = {}
            if 图像格式 == 'PNG':
                save_kwargs["pnginfo"] = metadata
                # 添加PNG压缩优化（默认开启优化）
                save_kwargs["compress_level"] = PNG压缩级别
                save_kwargs["optimize"] = True  # 默认开启PNG优化
            elif 图像格式 == 'JPG':
                save_kwargs["quality"] = 图像质量
                save_kwargs["optimize"] = True
                # JPG需要转换为RGB
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")
            elif 图像格式 == 'WEBP':
                save_kwargs["quality"] = 图像质量
                save_kwargs["lossless"] = WEBP无损
            
            # 保存图像（如果自动保存开启）
            if 自动保存:
                try:
                    img.save(save_path, **save_kwargs)
                    saved_files.append(save_path)
                    print(f"已保存图像: {save_path}")
                except Exception as e:
                    print(f"保存图像失败: {str(e)}")
            
            # 构建预览信息（除非关闭预览）
            if not 关闭预览:
                # 为预览创建临时文件
                preview_filename = f"preview_{filename_base}.png"
                preview_path = os.path.join(self.temp_dir, preview_filename)
                
                try:
                    # 保存预览图像（总是保存为PNG格式）
                    # 首先调整预览图像大小，使长边不超过1024像素
                    preview_img = self.resize_image_for_preview(img.copy(), max_size=1024)
                    
                    preview_metadata = PngImagePlugin.PngInfo()
                    if prompt is not None:
                        preview_metadata.add_text("prompt", json.dumps(prompt))
                    preview_metadata.add_text("generator", "MISLG AdvancedImageSaver Preview")
                    preview_metadata.add_text("preview_info", f"原始尺寸: {img.width}x{img.height}, 预览尺寸: {preview_img.width}x{preview_img.height}")
                    
                    # 预览图像优化加速：使用中等压缩级别（3），禁用优化以加快保存速度
                    preview_img.save(preview_path, pnginfo=preview_metadata, compress_level=3, optimize=False)
                    
                    # 添加预览信息到结果
                    results.append({
                        "filename": preview_filename,
                        "subfolder": "temp_previews",  # 明确指定子文件夹
                        "type": "output"
                    })
                    print(f"已创建预览: {preview_path} (尺寸: {preview_img.width}x{preview_img.height})")
                except Exception as e:
                    print(f"创建预览失败: {str(e)}")
        
        # 生成详细信息
        detail_info = []
        detail_info.append("=== 图像保存详情 ===")
        detail_info.append(f"保存目录: {output_path}")
        detail_info.append(f"图像格式: {图像格式}")
        
        # 根据格式显示不同的质量信息
        if 图像格式 == 'PNG':
            detail_info.append(f"PNG压缩级别: {PNG压缩级别} (0=不压缩, 9=最大压缩)")
            detail_info.append(f"PNG优化: 已开启")  # 固定显示已开启
        elif 图像格式 == 'JPG':
            detail_info.append(f"JPG图像质量: {图像质量}")
        elif 图像格式 == 'WEBP':
            detail_info.append(f"WEBP图像质量: {图像质量}")
            detail_info.append(f"WEBP无损: {'是' if WEBP无损 else '否'}")
        
        detail_info.append(f"日期目录: {'已添加' if 添加日期目录 else '未添加'}")
        detail_info.append(f"文件名日期: {'已添加' if 添加日期 else '未添加'}")
        detail_info.append(f"自动保存: {'已开启' if 自动保存 else '已关闭'}")
        detail_info.append(f"预览: {'已关闭' if 关闭预览 else '已开启'}")
        if not 关闭预览:
            detail_info.append(f"预览尺寸限制: 长边最大1024像素")
        
        if 自动保存:
            detail_info.append(f"保存数量: {len(saved_files)} 张图像")
            if saved_files:
                detail_info.append("\n=== 已保存文件 ===")
                for i, file_path in enumerate(saved_files):
                    detail_info.append(f"{i+1}. {os.path.basename(file_path)}")
        else:
            detail_info.append(f"预览数量: {len(图像)} 张图像")
            detail_info.append("注意: 自动保存已关闭，图像仅用于预览")
        
        # 返回UI信息和详细信息
        if 关闭预览:
            return ("\n".join(detail_info),)
        else:
            # 使用官方SaveImage的返回格式
            return {"ui": {"images": results}, "result": ("\n".join(detail_info),)}

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