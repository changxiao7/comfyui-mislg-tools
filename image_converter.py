"""
图像转换节点模块
将图像转换为各种像素格式，支持像素数据分析，并提供高级图像保存功能
"""

import torch
import numpy as np
import os
import time
import json
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "保存目录": (["默认输出", "自定义目录"], {"default": "默认输出"}),
                "文件名前缀": ("STRING", {"default": "ComfyUI"}),
                "图像格式": (["PNG", "JPG", "WEBP"], {"default": "PNG"}),
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
    
    def save_images(self, 图像, 保存目录, 文件名前缀, 图像格式, 图像质量, 添加日期目录, 添加日期, 自动保存, WEBP无损, 关闭预览,
                   自定义路径="", prompt=None, extra_pnginfo=None):
        """保存图像 - 基于官方实现优化"""
        
        # 检查是否启用自动保存
        if not 自动保存:
            return ("等待自动保存启用...",)
        
        # 确定输出目录
        if 保存目录 == "自定义目录" and 自定义路径.strip():
            output_path = 自定义路径.strip()
            # 创建自定义目录
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_dir
        
        # 添加日期目录
        if 添加日期目录:
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_path = os.path.join(output_path, date_str)
            os.makedirs(output_path, exist_ok=True)
        
        # 添加日期到文件名前缀
        if 添加日期:
            date_str = datetime.now().strftime("%Y%m%d")  # 只使用日期，去掉时间部分
            final_filename_prefix = f"{文件名前缀}_{date_str}"
        else:
            final_filename_prefix = 文件名前缀
        
        # 获取完整输出信息
        full_output_folder, filename, counter, subfolder, final_filename_prefix = (
            folder_paths.get_save_image_path(final_filename_prefix, output_path, 图像[0].shape[1], 图像[0].shape[0])
        )
        
        results = list()
        saved_files = []
        
        # 处理每张图像
        for image in 图像:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
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
            file = f"{filename}_{counter:05}_.{图像格式.lower()}"
            save_path = os.path.join(full_output_folder, file)
            
            save_kwargs = {}
            if 图像格式 == 'PNG':
                save_kwargs["pnginfo"] = metadata
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
            
            try:
                img.save(save_path, **save_kwargs)
                
                # 构建预览信息
                if not 关闭预览:
                    # 计算相对于输出目录的路径
                    rel_path = os.path.relpath(full_output_folder, self.output_dir)
                    if rel_path == ".":
                        rel_path = ""
                    
                    results.append({
                        "filename": file,
                        "subfolder": rel_path,
                        "type": self.type
                    })
                
                saved_files.append(save_path)
                counter += 1
            except Exception as e:
                print(f"保存图像失败: {str(e)}")
        
        # 生成详细信息
        detail_info = []
        detail_info.append("=== 图像保存详情 ===")
        detail_info.append(f"保存目录: {output_path}")
        detail_info.append(f"图像格式: {图像格式}")
        detail_info.append(f"图像质量: {图像质量}")
        detail_info.append(f"WEBP无损: {'是' if WEBP无损 else '否'}")
        detail_info.append(f"日期目录: {'已添加' if 添加日期目录 else '未添加'}")
        detail_info.append(f"文件名日期: {'已添加' if 添加日期 else '未添加'}")
        detail_info.append(f"预览: {'已关闭' if 关闭预览 else '已开启'}")
        detail_info.append(f"保存数量: {len(saved_files)} 张图像")
        
        if saved_files:
            detail_info.append("\n=== 已保存文件 ===")
            for i, file_path in enumerate(saved_files):
                detail_info.append(f"{i+1}. {os.path.basename(file_path)}")
        
        # 返回UI信息和详细信息
        if 关闭预览:
            return ("\n".join(detail_info),)
        else:
            # 确保返回格式正确，符合ComfyUI的预期
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