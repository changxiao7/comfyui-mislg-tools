"""
自定义路径图片加载器 - 集成即时预览和路径管理功能
作者: MISLG
"""

import os
import glob
import torch
from PIL import Image, ImageOps, ImageSequence
import numpy as np
import time
import folder_paths
import hashlib
import shutil
from datetime import datetime

class 自定义路径图片加载器:
    """
    自定义路径图片加载器 - 集成即时预览和路径管理功能
    """
    
    # 定义支持的图片格式
    SUPPORTED_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'tif', 'gif']
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        image_files = cls.get_image_files(input_dir)
        
        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        
        return {
            "required": {
                "图片文件": (image_files, {
                    "default": image_files[0] if image_files else "",
                    "image_upload": True  # 启用图片上传功能
                }),
                "操作模式": (["预览模式", "上传模式", "目录监控模式"], {
                    "default": "预览模式"
                }),
                "外部路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入文件路径(上传模式)或目录路径(监控模式)"
                }),
                "刷新控制": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "button",
                    "button_label": "🔄 刷新文件列表"
                }),
            },
            "optional": {
                "高级选项": ("BOOLEAN", {
                    "default": False,
                    "label_on": "🔧 显示高级选项",
                    "label_off": "🔧 隐藏高级选项"
                }),
                "自动刷新间隔": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 60,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "0表示禁用自动刷新"
                }),
                "图片预处理": (["无", "自动增强", "灰度化", "边缘检测"], {
                    "default": "无"
                }),
                "加载限制": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "目录监控模式下加载的文件数量限制"
                }),
                "缓存策略": (["智能缓存", "始终刷新", "禁用缓存"], {
                    "default": "智能缓存"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "状态信息")
    FUNCTION = "load_image_with_path_assistant"
    CATEGORY = "MISLG Tools/Image"
    DESCRIPTION = "自定义路径图片加载器 - 集成即时预览和路径管理功能"
    OUTPUT_NODE = True
    
    # 关键：使用这个特殊方法强制节点在输入变化时重新执行
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 返回随机值强制ComfyUI重新执行节点
        return float("NaN")
    
    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()
        self.output_dir = folder_paths.get_output_directory()
        self.last_filename = None
        self.cached_image = None
        self.cached_mask = None
        self.last_monitored_dir = None
        self.last_monitored_file = None
        self.file_list = []
        self.last_refresh_value = 0
        self.last_auto_refresh_time = 0

    @classmethod
    def get_image_files(cls, directory):
        """获取目录中的图片文件列表"""
        image_files = []
        try:
            # 使用硬编码的图片格式列表
            for ext in cls.SUPPORTED_IMAGE_EXTENSIONS:
                pattern = os.path.join(directory, f"*.{ext}")
                image_files.extend(glob.glob(pattern))
            
            # 按修改时间排序
            image_files.sort(key=os.path.getmtime, reverse=True)
            image_files = [os.path.basename(f) for f in image_files]
        except Exception as e:
            print(f"获取图片文件列表失败: {e}")
            image_files = []
        
        return image_files

    def load_image_with_path_assistant(self, 图片文件, 操作模式, 外部路径, 刷新控制=0, 高级选项=False, 自动刷新间隔=0, 图片预处理="无", 加载限制=10, 缓存策略="智能缓存"):
        状态信息 = ""
        
        # 检查是否需要刷新（按钮点击或自动刷新）
        current_time = time.time()
        need_refresh = False
        
        # 检查手动刷新按钮
        if 刷新控制 != self.last_refresh_value:
            need_refresh = True
            self.last_refresh_value = 刷新控制
            状态信息 += "🔄 手动刷新已触发\n"
        
        # 检查自动刷新
        if 自动刷新间隔 > 0 and (current_time - self.last_auto_refresh_time) >= 自动刷新间隔:
            need_refresh = True
            self.last_auto_refresh_time = current_time
            状态信息 += f"⏰ 自动刷新 ({自动刷新间隔}秒)\n"
        
        # 处理缓存策略
        if 缓存策略 == "始终刷新":
            need_refresh = True
            状态信息 += "💾 缓存策略: 始终刷新\n"
        elif 缓存策略 == "禁用缓存":
            self.cached_image = None
            self.cached_mask = None
            self.last_filename = None
            self.last_monitored_file = None
            状态信息 += "💾 缓存策略: 禁用缓存\n"
        else:
            状态信息 += "💾 缓存策略: 智能缓存\n"
        
        # 处理不同操作模式
        if 操作模式 == "上传模式":
            if 外部路径.strip():
                # 上传模式下，外部路径被解释为要上传的文件路径
                uploaded_file = self.upload_external_image(外部路径.strip())
                if uploaded_file:
                    图片文件 = uploaded_file
                    状态信息 += f"✅ 成功上传: {图片文件}\n"
                    print(f"图片上传成功: {图片文件}")
                    
                    # 上传成功后，加载新上传的图片
                    image_path = os.path.join(self.input_dir, 图片文件)
                    if os.path.exists(image_path):
                        try:
                            image, mask = self.load_image(image_path)
                            
                            # 应用图片预处理
                            if 图片预处理 != "无":
                                image = self.apply_image_preprocessing(image, 图片预处理)
                                状态信息 += f"🛠️ 已应用预处理: {图片预处理}\n"
                            
                            # 更新缓存
                            if 缓存策略 != "禁用缓存":
                                self.last_filename = 图片文件
                                self.cached_image = image
                                self.cached_mask = mask
                            
                            # 获取图片信息
                            img_info = self.get_image_info(image_path)
                            状态信息 += f"✅ 成功加载: {图片文件}\n{img_info}"
                            
                            return (image, mask, 图片文件, 状态信息)
                            
                        except Exception as e:
                            return self.create_empty_output(f"加载上传的图片失败: {e}")
                    else:
                        return self.create_empty_output("上传的图片文件不存在")
                else:
                    return self.create_empty_output("文件上传失败")
            else:
                return self.create_empty_output("请提供要上传的文件路径")
        
        elif 操作模式 == "目录监控模式":
            if 外部路径.strip():
                # 检查是否需要刷新文件列表
                if need_refresh or not self.file_list or 外部路径 != self.last_monitored_dir:
                    self.file_list = self.get_directory_files(外部路径.strip(), 加载限制)
                    self.last_monitored_dir = 外部路径.strip()
                    状态信息 += f"🔄 目录文件列表已刷新 (限制: {加载限制}个文件)\n"
                
                if not self.file_list:
                    return self.create_empty_output("监控目录中没有图片文件")
                
                # 获取最新文件
                latest_file = self.file_list[0] if self.file_list else None
                if latest_file:
                    # 检查是否需要重新加载
                    if (latest_file == self.last_monitored_file and 
                        self.cached_image is not None and 
                        self.cached_mask is not None and
                        not need_refresh and
                        缓存策略 != "始终刷新"):
                        状态信息 += f"使用缓存图片: {os.path.basename(latest_file)}\n"
                        return (self.cached_image, self.cached_mask, os.path.basename(latest_file), 状态信息)
                    
                    # 加载最新图片
                    image, mask, info = self.load_external_image(latest_file)
                    if image is not None:
                        # 应用图片预处理
                        if 图片预处理 != "无":
                            image = self.apply_image_preprocessing(image, 图片预处理)
                            状态信息 += f"🛠️ 已应用预处理: {图片预处理}\n"
                        
                        # 更新缓存
                        if 缓存策略 != "禁用缓存":
                            self.last_monitored_file = latest_file
                            self.cached_image = image
                            self.cached_mask = mask
                        
                        状态信息 += f"✅ 已加载最新图片: {os.path.basename(latest_file)}\n{info}"
                        
                        return (image, mask, os.path.basename(latest_file), 状态信息)
                    else:
                        return self.create_empty_output(f"无法加载图片: {latest_file}")
                else:
                    return self.create_empty_output("没有找到可用的图片文件")
            else:
                return self.create_empty_output("请提供要监控的目录路径")
        
        # 预览模式 - 从输入目录加载图片
        if not 图片文件:
            return self.create_empty_output("未选择图片文件")
        
        image_path = os.path.join(self.input_dir, 图片文件)
        
        if not os.path.exists(image_path):
            return self.create_empty_output(f"图片文件不存在: {图片文件}")
        
        # 检查是否需要重新加载
        if 图片文件 == self.last_filename and self.cached_image is not None and not need_refresh and 缓存策略 != "始终刷新":
            状态信息 += "使用缓存图片\n"
            return (self.cached_image, self.cached_mask, 图片文件, 状态信息)
        
        # 加载图片
        try:
            image, mask = self.load_image(image_path)
            
            # 应用图片预处理
            if 图片预处理 != "无":
                image = self.apply_image_preprocessing(image, 图片预处理)
                状态信息 += f"🛠️ 已应用预处理: {图片预处理}\n"
            
            # 更新缓存
            if 缓存策略 != "禁用缓存":
                self.last_filename = 图片文件
                self.cached_image = image
                self.cached_mask = mask
            
            # 获取图片信息
            img_info = self.get_image_info(image_path)
            状态信息 += f"✅ 成功加载: {图片文件}\n{img_info}"
            
            print(f"即时预览图片加载器: 已加载 {图片文件}")
            
            return (image, mask, 图片文件, 状态信息)
            
        except Exception as e:
            return self.create_empty_output(f"加载图片失败: {e}")
    
    def get_directory_files(self, directory_path, limit=10):
        """获取目录中的所有图片文件，按修改时间排序"""
        try:
            if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                print(f"目录不存在或不是目录: {directory_path}")
                return []
            
            image_files = []
            # 使用硬编码的图片格式列表
            for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                pattern = os.path.join(directory_path, f"*.{ext}")
                image_files.extend(glob.glob(pattern))
            
            # 按修改时间排序（最新的在前）
            image_files.sort(key=os.path.getmtime, reverse=True)
            
            # 应用加载限制
            if limit > 0 and len(image_files) > limit:
                image_files = image_files[:limit]
            
            print(f"在目录 {directory_path} 中找到 {len(image_files)} 个图片文件 (限制: {limit})")
            return image_files
            
        except Exception as e:
            print(f"获取目录文件列表失败: {e}")
            return []
    
    def apply_image_preprocessing(self, image_tensor, preprocessing_type):
        """应用图片预处理"""
        try:
            if preprocessing_type == "无":
                return image_tensor
                
            # 将tensor转换为numpy数组进行处理
            image_array = image_tensor[0].numpy()
            
            if preprocessing_type == "自动增强":
                # 简单的自动对比度增强
                min_val = np.min(image_array)
                max_val = np.max(image_array)
                if max_val > min_val:
                    image_array = (image_array - min_val) / (max_val - min_val)
            
            elif preprocessing_type == "灰度化":
                # 转换为灰度图像
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
                image_array = np.stack([gray, gray, gray], axis=-1)
            
            elif preprocessing_type == "边缘检测":
                # 简单的Sobel边缘检测
                from scipy import ndimage
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
                
                # Sobel算子
                sobel_x = ndimage.sobel(gray, axis=1)
                sobel_y = ndimage.sobel(gray, axis=0)
                edges = np.hypot(sobel_x, sobel_y)
                edges = edges / np.max(edges) if np.max(edges) > 0 else edges
                
                image_array = np.stack([edges, edges, edges], axis=-1)
            
            # 转换回tensor
            processed_tensor = torch.from_numpy(image_array).unsqueeze(0)
            return processed_tensor
            
        except Exception as e:
            print(f"图片预处理失败: {e}")
            return image_tensor
    
    def upload_external_image(self, source_path):
        """上传外部图片到输入目录"""
        try:
            if not os.path.exists(source_path):
                print(f"源文件不存在: {source_path}")
                return None
            
            # 验证是否为图片文件
            try:
                with Image.open(source_path) as img:
                    img.verify()
            except Exception as e:
                print(f"文件不是有效的图片格式: {source_path}, 错误: {e}")
                return None
            
            # 获取文件名
            filename = os.path.basename(source_path)
            target_path = os.path.join(self.input_dir, filename)
            
            # 处理文件名冲突
            counter = 1
            name, ext = os.path.splitext(filename)
            while os.path.exists(target_path):
                new_filename = f"{name}_{counter}{ext}"
                target_path = os.path.join(self.input_dir, new_filename)
                counter += 1
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            print(f"图片上传成功: {source_path} -> {target_path}")
            
            return os.path.basename(target_path)
            
        except Exception as e:
            print(f"图片上传失败: {e}")
            return None
    
    def load_external_image(self, image_path):
        """直接加载外部图片，不上传到输入目录"""
        try:
            if not os.path.exists(image_path):
                return None, None, "文件不存在"
            
            # 验证是否为图片文件
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                return None, None, f"不是有效的图片格式: {e}"
            
            # 加载图片
            image, mask = self.load_image(image_path)
            
            # 获取图片信息
            img_info = self.get_image_info(image_path)
            
            return image, mask, img_info
            
        except Exception as e:
            return None, None, f"加载失败: {e}"
    
    def load_image(self, image_path):
        """加载图片文件 - 修复版本"""
        try:
            # 使用 ComfyUI 的标准方法加载图片
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            
            # 处理图片模式
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            
            # 转换为numpy数组
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            
            # 处理mask
            if hasattr(i, 'getchannel') and 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = 1.0 - torch.from_numpy(mask)
            else:
                # 创建全白mask
                mask_tensor = torch.zeros((image_array.shape[0], image_array.shape[1]), dtype=torch.float32)
                mask_tensor = mask_tensor.unsqueeze(0)
            
            return (image_tensor, mask_tensor)
            
        except Exception as e:
            print(f"加载图片失败: {e}")
            # 如果标准方法失败，使用备用方法
            try:
                return self.load_image_fallback(image_path)
            except Exception as e2:
                print(f"备用加载方法也失败: {e2}")
                raise e
    
    def load_image_fallback(self, image_path):
        """备用图片加载方法"""
        image = Image.open(image_path)
        
        # 转换为RGB模式
        if image.mode == 'RGBA':
            rgb_image = image.convert('RGB')
            # 提取alpha通道作为mask
            alpha_mask = image.split()[-1]
        else:
            rgb_image = image.convert('RGB')
            alpha_mask = None
        
        # 转换为numpy数组
        image_array = np.array(rgb_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)[None,]
        
        # 处理mask
        if alpha_mask is not None:
            mask_array = np.array(alpha_mask).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask_array)[None,]
        else:
            # 创建全白mask (没有透明通道)
            mask_tensor = torch.zeros((1, image_array.shape[0], image_array.shape[1]), dtype=torch.float32)
        
        return image_tensor, mask_tensor
    
    def get_image_info(self, image_path):
        """获取图片信息"""
        try:
            with Image.open(image_path) as img:
                dimensions = img.size
                mode = img.mode
                format_info = img.format
            
            file_size = os.path.getsize(image_path)
            mod_time = time.ctime(os.path.getmtime(image_path))
            
            info = f"尺寸: {dimensions[0]}x{dimensions[1]}\n"
            info += f"模式: {mode}\n"
            info += f"格式: {format_info}\n"
            info += f"大小: {self.format_file_size(file_size)}\n"
            info += f"修改: {mod_time}"
            
            return info
        except Exception as e:
            return f"获取图片信息失败: {e}"
    
    def format_file_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names)-1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    def create_empty_output(self, error_message):
        """创建空输出"""
        # 创建默认的黑色图像和白色mask
        empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)  # 白色mask
        print(f"即时预览图片加载器错误: {error_message}")
        return (empty_image, empty_mask, "加载失败", error_message)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "InstantPreviewImageLoaderWithPath": 自定义路径图片加载器,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantPreviewImageLoaderWithPath": "自定义路径图片加载器",
}