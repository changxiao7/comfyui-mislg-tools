"""
图像转换节点模块
将图像转换为各种像素格式，支持像素数据分析，并提供高级图像保存功能
"""

import torch
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image
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
    CATEGORY = "MISLG Tools/Image"
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
    CATEGORY = "MISLG Tools/Image"
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
    """高级图像保存器 - 支持多目录选择和多种格式保存，自动保存"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "保存目录": (["默认输出目录", "自定义目录"], {
                    "default": "默认输出目录"
                }),
                "图像格式": (["PNG", "JPG", "WEBP", "BMP", "TIFF"], {
                    "default": "PNG"
                }),
                "文件名前缀": ("STRING", {
                    "default": "image",
                    "multiline": False,
                    "placeholder": "输入文件名前缀"
                }),
                "启用日期目录": ("BOOLEAN", {
                    "default": True,
                    "label_on": "✅ 启用",
                    "label_off": "❌ 禁用"
                }),
                "自动保存": ("BOOLEAN", {
                    "default": True,
                    "label_on": "✅ 自动保存",
                    "label_off": "❌ 手动保存"
                }),
            },
            "optional": {
                "自定义目录路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入自定义目录路径"
                }),
                "质量设置": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "JPG/WEBP格式的图像质量(1-100)"
                }),
                "嵌入工作流": ("BOOLEAN", {
                    "default": True,
                    "label_on": "✅ 嵌入",
                    "label_off": "❌ 不嵌入"
                }),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "输入提示词信息"
                }),
                "负面提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "输入负面提示词信息"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("详细信息",)
    FUNCTION = "save_images"
    CATEGORY = "MISLG Tools/Image"
    OUTPUT_NODE = True
    DESCRIPTION = "高级图像保存器 - 支持多目录选择和多种格式保存，自动保存，支持提示词嵌入"
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    def save_images(self, images, 保存目录, 图像格式, 文件名前缀, 启用日期目录, 自动保存, 自定义目录路径="", 质量设置=95, 嵌入工作流=True, 提示词="", 负面提示词=""):
        """保存图像到指定目录"""
        
        # 检查是否启用自动保存
        if not 自动保存:
            return ("等待自动保存启用...",)
        
        if images is None or len(images) == 0:
            return ("❌ 错误: 没有输入图像",)
        
        # 确定保存目录
        if 保存目录 == "默认输出目录":
            base_dir = self.output_dir
        else:
            if not 自定义目录路径.strip():
                return ("❌ 错误: 自定义目录路径为空",)
            base_dir = 自定义目录路径.strip()
        
        # 添加日期子目录
        if 启用日期目录:
            date_str = datetime.now().strftime("%Y-%m-%d")
            save_dir = os.path.join(base_dir, date_str)
        else:
            save_dir = base_dir
        
        # 确保目录存在
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            return (f"❌ 目录创建失败: {e}\n目录路径: {save_dir}",)
        
        # 获取下一个序列号
        next_number = self.get_next_sequence_number(save_dir, 文件名前缀, 图像格式.lower())
        
        # 保存图像
        saved_files = []
        total_images = len(images)
        error_messages = []
        
        for i, image in enumerate(images):
            # 生成文件名，基于序列号
            file_number = next_number + i
            filename = f"{文件名前缀}_{file_number:05d}.{图像格式.lower()}"
            
            file_path = os.path.join(save_dir, filename)
            
            try:
                # 转换图像格式
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                
                # 根据格式保存
                save_params = {}
                
                if 图像格式 == "PNG":
                    # PNG格式特殊处理，支持嵌入工作流和提示词
                    if 嵌入工作流:
                        # 获取工作流信息
                        workflow_info = self.get_workflow_info(提示词, 负面提示词)
                        if workflow_info:
                            pil_image.save(file_path, format="PNG", pnginfo=workflow_info)
                        else:
                            pil_image.save(file_path, format="PNG")
                    else:
                        pil_image.save(file_path, format="PNG")
                
                elif 图像格式 == "JPG":
                    save_params["quality"] = 质量设置
                    save_params["optimize"] = True
                    # 转换为RGB（JPG不支持透明通道）
                    if pil_image.mode in ("RGBA", "LA"):
                        background = Image.new("RGB", pil_image.size, (255, 255, 255))
                        background.paste(pil_image, mask=pil_image.split()[-1])
                        pil_image = background
                    elif pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    
                    # 为JPG添加EXIF数据（包含提示词）
                    exif_data = self.get_exif_data(提示词, 负面提示词)
                    if exif_data:
                        pil_image.save(file_path, format="JPEG", exif=exif_data, **save_params)
                    else:
                        pil_image.save(file_path, format="JPEG", **save_params)
                
                elif 图像格式 == "WEBP":
                    save_params["quality"] = 质量设置
                    # 为WEBP添加元数据
                    if 嵌入工作流:
                        exif_data = self.get_exif_data(提示词, 负面提示词)
                        if exif_data:
                            pil_image.save(file_path, format="WEBP", exif=exif_data, **save_params)
                        else:
                            pil_image.save(file_path, format="WEBP", **save_params)
                    else:
                        pil_image.save(file_path, format="WEBP", **save_params)
                
                elif 图像格式 == "BMP":
                    pil_image.save(file_path, format="BMP")
                
                elif 图像格式 == "TIFF":
                    # TIFF格式也支持元数据
                    if 嵌入工作流:
                        exif_data = self.get_exif_data(提示词, 负面提示词)
                        if exif_data:
                            pil_image.save(file_path, format="TIFF", exif=exif_data)
                        else:
                            pil_image.save(file_path, format="TIFF")
                    else:
                        pil_image.save(file_path, format="TIFF")
                
                saved_files.append(file_path)
                print(f"图像保存成功: {file_path}")
                
            except Exception as e:
                error_msg = f"❌ 图像 {i+1} 保存失败: {e}"
                print(error_msg)
                error_messages.append(error_msg)
        
        # 生成详细信息
        details = []
        
        if error_messages:
            # 有错误发生
            details.extend(error_messages)
            if saved_files:
                details.append(f"部分保存成功: {len(saved_files)} 张图像")
                for file_path in saved_files:
                    details.append(f"✅ 已保存: {os.path.basename(file_path)}")
        elif saved_files:
            # 全部保存成功
            if len(saved_files) == 1:
                details.append(f"✅ 图像保存成功: {os.path.basename(saved_files[0])}")
            else:
                details.append(f"✅ 批量保存成功: {len(saved_files)} 张图像")
            
            # 添加文件路径信息
            details.append(f"保存目录: {save_dir}")
            for file_path in saved_files:
                details.append(f"📄 {os.path.basename(file_path)}")
        else:
            # 没有保存任何文件
            details.append("❌ 没有保存任何图像")
        
        # 添加配置信息
        details.append(f"格式: {图像格式} | 前缀: {文件名前缀} | 起始序号: {next_number:05d}")
        
        # 添加提示词信息
        if 提示词:
            details.append(f"提示词: {提示词[:50]}{'...' if len(提示词) > 50 else ''}")
        if 负面提示词:
            details.append(f"负面提示词: {负面提示词[:50]}{'...' if len(负面提示词) > 50 else ''}")
        
        # 添加工作流嵌入信息
        if 嵌入工作流 and 图像格式 in ["PNG", "JPG", "WEBP", "TIFF"]:
            details.append("工作流信息: 已嵌入")
        
        # 添加质量设置信息
        if 图像格式 in ["JPG", "WEBP"]:
            details.append(f"质量设置: {质量设置}")
        
        return ("\n".join(details),)
    
    def get_next_sequence_number(self, directory, prefix, extension):
        """获取目录中下一个可用的序列号"""
        try:
            # 检查目录是否存在
            if not os.path.exists(directory):
                return 1
            
            # 获取目录中所有匹配的文件
            pattern = f"{prefix}_*.{extension}"
            existing_files = []
            
            for file in os.listdir(directory):
                if file.startswith(f"{prefix}_") and file.endswith(f".{extension}"):
                    existing_files.append(file)
            
            # 如果没有文件，从00001开始
            if not existing_files:
                return 1
            
            # 提取所有序号
            numbers = []
            for file in existing_files:
                try:
                    # 提取文件名中的数字部分
                    # 文件名格式: prefix_XXXXX.extension
                    base_name = os.path.splitext(file)[0]  # 去掉扩展名
                    num_part = base_name[len(prefix)+1:]  # 去掉前缀和_
                    
                    # 确保是5位数字
                    if num_part.isdigit() and len(num_part) == 5:
                        numbers.append(int(num_part))
                except Exception as e:
                    print(f"解析文件名 {file} 失败: {e}")
                    continue
            
            # 如果没有找到有效数字，从00001开始
            if not numbers:
                return 1
            
            # 返回最大序号+1
            return max(numbers) + 1
            
        except Exception as e:
            print(f"获取序列号失败: {e}")
            return 1
    
    def get_workflow_info(self, prompt="", negative_prompt=""):
        """获取工作流信息 - 包含提示词"""
        try:
            # 使用PIL的PngInfo类
            from PIL.PngImagePlugin import PngInfo
            
            # 创建PNG信息对象
            pnginfo = PngInfo()
            
            # 添加基本的工作流信息
            pnginfo.add_text("Software", "ComfyUI MISLG Tools")
            pnginfo.add_text("CreationTime", datetime.now().isoformat())
            
            # 添加提示词信息
            if prompt:
                pnginfo.add_text("Prompt", prompt)
            
            if negative_prompt:
                pnginfo.add_text("NegativePrompt", negative_prompt)
            
            # 添加其他有用的元数据
            pnginfo.add_text("Generator", "AdvancedImageSaver")
            pnginfo.add_text("Parameters", f"Quality: 95, Workflow: Embedded")
            
            return pnginfo
            
        except Exception as e:
            print(f"获取工作流信息失败: {e}")
            return None
    
    def get_exif_data(self, prompt="", negative_prompt=""):
        """为JPG/WEBP/TIFF格式创建EXIF数据"""
        try:
            # 创建空的EXIF数据
            exif_dict = {}
            
            # 添加软件信息
            exif_dict[0x013b] = "ComfyUI MISLG Tools"  # Artist
            exif_dict[0x010e] = "Generated by AdvancedImageSaver"  # ImageDescription
            
            # 添加提示词到EXIF注释
            comment_parts = []
            if prompt:
                # 截断过长的提示词
                truncated_prompt = prompt[:1000] + "..." if len(prompt) > 1000 else prompt
                comment_parts.append(f"Prompt: {truncated_prompt}")
            
            if negative_prompt:
                # 截断过长的负面提示词
                truncated_negative = negative_prompt[:500] + "..." if len(negative_prompt) > 500 else negative_prompt
                comment_parts.append(f"Negative: {truncated_negative}")
            
            if comment_parts:
                exif_dict[0x9286] = " | ".join(comment_parts)  # UserComment
            
            # 添加创建时间
            from PIL.ExifTags import TAGS
            exif_dict[0x9003] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")  # DateTimeOriginal
            
            # 将EXIF字典转换为字节
            from PIL import Image
            exif_bytes = Image.Exif()
            for tag, value in exif_dict.items():
                exif_bytes[tag] = value
            
            return exif_bytes.tobytes()
            
        except Exception as e:
            print(f"创建EXIF数据失败: {e}")
            return None

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