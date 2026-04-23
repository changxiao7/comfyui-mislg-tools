"""
即时预览图片加载器 - 基于官方 LoadImage 的增强版
支持自定义路径上传、目录监控、智能缓存与外部遮罩透传
作者: MISLG | 修复: 语法/空格/缩进/ComfyUI兼容性优化
"""
import os
import glob
import torch
from PIL import Image, ImageOps
import numpy as np
import time
import folder_paths
import shutil
import hashlib

# 兼容新旧版 Pillow
_RESAMPLE = getattr(Image, 'Resampling', Image).BILINEAR

class InstantPreviewImageLoader:
    """增强版图片加载器 - 支持外部路径和目录监控"""
    SUPPORTED_EXT = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif']

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        if os.path.isdir(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            files = [f for f in files if any(f.lower().endswith(ext) for ext in cls.SUPPORTED_EXT)]
        else:
            files = []

        return {
            "required": {
                "图片文件": (sorted(files), {"image_upload": True}),
                "操作模式": (["预览模式", "上传模式", "目录监控模式"], {"default": "预览模式"}),
                "外部路径": ("STRING", {"default": "", "multiline": False, "placeholder": "文件路径(上传模式)或目录路径(监控模式)"}),
                "点击刷新": ("BOOLEAN", {"default": False, "label_on": "强制刷新", "label_off": "使用缓存"}),
                "加载限制": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1, "tooltip": "目录监控模式下加载的文件数量限制"}),
                "缓存策略": (["智能缓存", "始终刷新", "禁用缓存"], {"default": "智能缓存"}),
            },
            "optional": {
                "外部遮罩输入": ("MASK",),
                "遮罩操作": (["使用外部遮罩", "覆盖外部遮罩", "忽略外部遮罩"], {"default": "使用外部遮罩"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "status_info")
    FUNCTION = "load_image"
    CATEGORY = "MISLG Tools/Image"
    OUTPUT_NODE = True

    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()
        self.cache = {}
        self.monitor_cache = {}   # 存储目录文件列表
        self.last_refresh = False
        self.monitor_last_file = None
        self.monitor_last_time = 0
        self.monitor_last_size = 0

    def load_image(self, 图片文件, 操作模式, 外部路径, 点击刷新=False,
                   加载限制=10, 缓存策略="智能缓存", 外部遮罩输入=None, 遮罩操作="使用外部遮罩", unique_id=None):
        
        status_info = []
        current_time = time.time()
        
        # 处理手动强制刷新
        force_refresh = False
        if 点击刷新 != self.last_refresh:
            force_refresh = True
            self.last_refresh = 点击刷新
            status_info.append("🖱️ 强制刷新已触发")
        
        # 缓存策略处理
        if 缓存策略 == "始终刷新":
            force_refresh = True
            status_info.append("🔄 缓存策略：始终刷新")
        elif 缓存策略 == "禁用缓存":
            self.cache.clear()
            self.monitor_cache.clear()
            force_refresh = True
            status_info.append("🗑️ 缓存已清空")
        
        if 操作模式 == "上传模式":
            return self._handle_upload_mode(外部路径, 缓存策略, 外部遮罩输入, 遮罩操作, status_info, force_refresh)
        
        elif 操作模式 == "目录监控模式":
            return self._handle_monitor_mode(外部路径, 加载限制, 缓存策略, 外部遮罩输入,
                                           遮罩操作, status_info, force_refresh, current_time)
        
        else:  # 预览模式
            return self._handle_preview_mode(图片文件, 缓存策略, 外部遮罩输入, 遮罩操作, status_info, force_refresh)

    def _handle_upload_mode(self, external_path, cache_policy, external_mask, 
                            mask_operation, status_info, force_refresh):
        ext_path = external_path.strip()
        if not ext_path:
            return self._create_empty_output("请提供要上传的文件路径")
        
        # 将相对路径转为绝对路径（基于 ComfyUI 根目录）
        if not os.path.isabs(ext_path):
            ext_path = os.path.join(folder_paths.base_path, ext_path)
        
        is_valid, validation_msg = self._validate_external_path(ext_path, "upload")
        status_info.append(validation_msg)
        if not is_valid:
            return self._create_empty_output(f"路径验证失败: {validation_msg}")
        
        uploaded_file = self._upload_external_image(ext_path)
        if not uploaded_file:
            return self._create_empty_output("文件上传失败")
        
        status_info.append(f"✅ 成功上传: {uploaded_file}")
        image_path = os.path.join(self.input_dir, uploaded_file)
        return self._load_and_process_image(image_path, uploaded_file, cache_policy, 
                                           external_mask, mask_operation, status_info, force_refresh)

    def _handle_monitor_mode(self, external_path, load_limit, cache_policy, external_mask,
                             mask_operation, status_info, force_refresh, current_time):
        ext_path = external_path.strip()
        if not ext_path:
            return self._create_empty_output("请提供要监控的目录路径")
        
        # 将相对路径转为绝对路径（基于 ComfyUI 根目录）
        if not os.path.isabs(ext_path):
            ext_path = os.path.join(folder_paths.base_path, ext_path)
        
        is_valid, validation_msg = self._validate_external_path(ext_path, "monitor")
        status_info.append(validation_msg)
        if not is_valid:
            return self._create_empty_output(f"路径验证失败: {validation_msg}")
        
        # 监控模式下每次执行都重新扫描目录（获取最新文件列表）
        files = self._get_directory_files(ext_path, load_limit)
        if not files:
            return self._create_empty_output("监控目录中没有图片文件")
        
        # 取修改时间最新的文件
        latest_file = files[0]
        cache_key = f"monitor_{os.path.basename(latest_file)}"
        
        # 检查文件是否变化（新文件或内容更新）
        file_changed = latest_file != self.monitor_last_file
        file_updated = self._is_file_updated(latest_file, current_time)
        
        # 决定是否需要重新加载
        need_reload = (force_refresh or file_changed or file_updated or
                       cache_key not in self.cache)
        
        if need_reload:
            status_info.append(f"📂 扫描目录: {ext_path}")
            image, mask, file_info = self._load_external_image(latest_file)
            if image is None:
                return self._create_empty_output(f"无法加载图片: {latest_file}")
            
            final_mask = self._process_external_mask(mask, external_mask, mask_operation)
            status_info.append(self._get_mask_status(external_mask, mask_operation))
            
            if cache_policy != "禁用缓存":
                self.cache[cache_key] = {'image': image, 'mask': final_mask}
            
            self.monitor_last_file = latest_file
            self.monitor_last_time = current_time
            try:
                self.monitor_last_size = os.path.getsize(latest_file)
            except:
                self.monitor_last_size = 0
            
            status_info.append(f"✅ 已加载: {os.path.basename(latest_file)}")
            status_info.append(file_info)
            return (image, final_mask, "\n".join(status_info))
        else:
            status_info.append(f"💾 使用缓存: {os.path.basename(latest_file)}")
            cached = self.cache[cache_key]
            return (cached['image'], cached['mask'], "\n".join(status_info))

    def _is_file_updated(self, file_path, current_time):
        """检测文件是否被更新（比较修改时间和大小）"""
        try:
            if not os.path.exists(file_path):
                return False
            mod_time = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            # 如果修改时间或大小发生变化，认为有更新
            if mod_time > self.monitor_last_time or size != self.monitor_last_size:
                return True
            return False
        except:
            return False

    def _handle_preview_mode(self, image_name, cache_policy, external_mask, mask_operation,
                             status_info, force_refresh):
        if not image_name:
            return self._create_empty_output("未选择图片文件")
        
        image_path = folder_paths.get_annotated_filepath(image_name)
        if not os.path.exists(image_path):
            return self._create_empty_output(f"图片文件不存在: {image_name}")
        
        cache_key = f"preview_{image_name}"
        if cache_key in self.cache and not force_refresh and cache_policy != "始终刷新":
            status_info.append("💾 使用缓存图片")
            cached = self.cache[cache_key]
            return (cached['image'], cached['mask'], "\n".join(status_info))
        
        return self._load_and_process_image(image_path, image_name, cache_policy, 
                                          external_mask, mask_operation, status_info, force_refresh)

    def _load_and_process_image(self, image_path, image_name, cache_policy, external_mask, 
                                mask_operation, status_info, force_refresh):
        try:
            image, mask = self._load_image_official_compatible(image_path)
            final_mask = self._process_external_mask(mask, external_mask, mask_operation)
            status_info.append(self._get_mask_status(external_mask, mask_operation))
            
            cache_key = f"preview_{image_name}"
            if cache_policy != "禁用缓存":
                self.cache[cache_key] = {'image': image, 'mask': final_mask}
            
            status_info.append(f"✅ 成功加载: {image_name}")
            status_info.append(self._get_image_info(image_path))
            return (image, final_mask, "\n".join(status_info))
        except Exception as e:
            return self._create_empty_output(f"加载图片失败: {str(e)}")

    def _load_image_official_compatible(self, image_path):
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                if len(image.shape) == 4:
                    mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32)
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32)
            
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            return (image, mask)
        except Exception as e:
            print(f"主加载方法失败: {e}")
            return self._load_image_fallback(image_path)

    def _load_image_fallback(self, image_path):
        image = Image.open(image_path)
        rgb_image = image.convert('RGB')
        image_array = np.array(rgb_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)[None,]
        mask_tensor = self._generate_mask_fallback(image, image_tensor.shape)
        return image_tensor, mask_tensor

    def _generate_mask_fallback(self, image, image_shape):
        h, w = image_shape[1], image_shape[2]
        if hasattr(image, 'getbands') and 'A' in image.getbands():
            try:
                mask_array = np.array(image.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = 1.0 - torch.from_numpy(mask_array)
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                return mask_tensor
            except:
                return torch.zeros((1, h, w), dtype=torch.float32)
        return torch.zeros((1, h, w), dtype=torch.float32)

    def _process_external_mask(self, original_mask, external_mask, mask_operation):
        if external_mask is None:
            return original_mask
        
        if len(external_mask.shape) == 2:
            processed_external_mask = external_mask.unsqueeze(0)
        elif len(external_mask.shape) == 3:
            processed_external_mask = external_mask
        else:
            return original_mask
            
        if mask_operation in ["使用外部遮罩", "覆盖外部遮罩"]:
            return processed_external_mask
        return original_mask

    def _get_mask_status(self, external_mask, mask_operation):
        if external_mask is not None:
            return f"🎭 遮罩模式: {mask_operation}"
        return "🎭 使用原始遮罩"

    def _validate_external_path(self, path, mode):
        p = path.strip()
        if not p:
            return False, "路径不能为空"
        
        if mode == "upload":
            if not os.path.exists(p):
                return False, f"文件不存在: {p}"
            if not os.path.isfile(p):
                return False, f"路径不是文件: {p}"
            ext = os.path.splitext(p)[1].lower()
            if ext not in self.SUPPORTED_EXT:
                return False, f"不支持的格式: {ext}"
            return True, f"✅ 文件有效: {os.path.basename(p)}"
        
        elif mode == "monitor":
            if not os.path.exists(p):
                return False, f"目录不存在: {p}"
            if not os.path.isdir(p):
                return False, f"路径不是目录: {p}"
            return True, f"✅ 目录有效: {p}"
        
        return False, "未知模式"

    def _upload_external_image(self, source_path):
        try:
            if not os.path.exists(source_path):
                return None
            with Image.open(source_path) as img:
                img.verify()
            
            filename = os.path.basename(source_path)
            target_path = os.path.join(self.input_dir, filename)
            
            counter = 1
            name, ext = os.path.splitext(filename)
            while os.path.exists(target_path):
                new_filename = f"{name}_{counter}{ext}"
                target_path = os.path.join(self.input_dir, new_filename)
                counter += 1
            
            shutil.copy2(source_path, target_path)
            return os.path.basename(target_path)
        except Exception as e:
            print(f"图片上传失败: {e}")
            return None

    def _load_external_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                return None, None, "文件不存在"
            image, mask = self._load_image_official_compatible(image_path)
            return image, mask, self._get_image_info(image_path)
        except Exception as e:
            return None, None, f"加载失败: {e}"

    def _get_directory_files(self, directory_path, limit=10):
        try:
            if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                return []
            files = []
            for ext in self.SUPPORTED_EXT:
                files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            files.sort(key=os.path.getmtime, reverse=True)
            return files[:limit] if limit > 0 else files
        except Exception as e:
            print(f"获取目录文件失败: {e}")
            return []

    def _get_image_info(self, image_path):
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                return f"📐 尺寸: {w}x{h} | 模式: {img.mode} | 大小: {self._format_file_size(os.path.getsize(image_path))}"
        except:
            return "📐 信息获取失败"

    def _format_file_size(self, size_bytes):
        if size_bytes == 0: return "0 B"
        units = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.2f} {units[i]}"

    def _create_empty_output(self, error_message):
        print(f"即时预览加载器错误: {error_message}")
        empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
        return (empty_image, empty_mask, error_message)

    @classmethod
    def IS_CHANGED(cls, 图片文件, **kwargs):
        image_path = folder_paths.get_annotated_filepath(图片文件)
        if not os.path.exists(image_path):
            return float("inf")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "InstantPreviewImageLoader": InstantPreviewImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantPreviewImageLoader": "🖼️ 即时预览图片加载器 (MISLG)",
}