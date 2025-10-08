"""
自定义路径图片加载器 - 基于官方节点的优化版本（中文界面）
整合官方LoadImage节点的核心功能，添加自定义路径和监控功能
作者: MISLG
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

class InstantPreviewImageLoader:
    """
    基于官方LoadImage节点的自定义图片加载器
    保留官方节点核心功能，添加路径管理和监控功能
    """
    
    # 定义支持的图片格式（与官方保持一致）
    SUPPORTED_EXT = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif']
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = [f for f in files if any(f.lower().endswith(ext) for ext in s.SUPPORTED_EXT)]
        
        return {
            "required": {
                "图片文件": (sorted(files), {"image_upload": True}),
                "操作模式": (["预览模式", "上传模式", "目录监控模式"], {"default": "预览模式"}),
                "外部路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "文件路径(上传模式)或目录路径(监控模式)"
                }),
                "刷新控制": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "button",
                    "button_label": "🔄 刷新文件列表"
                }),
                "自动刷新间隔": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 300,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "0 = 禁用自动刷新，单位：秒"
                }),
                "加载限制": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "目录监控模式下加载的文件数量限制"
                }),
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
    RETURN_NAMES = ("image", "mask", "状态信息")
    FUNCTION = "load_image"
    CATEGORY = "MISLG Tools/Image"
    DESCRIPTION = "增强版图片加载器 - 支持外部路径和目录监控"
    OUTPUT_NODE = True

    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()
        self.output_dir = folder_paths.get_output_directory()
        self.cache = {}
        self.monitor_cache = {}
        self.last_refresh = 0
        self.last_auto_refresh = 0
        self.monitor_last_file = None
        self.monitor_last_time = 0

    def load_image(self, 图片文件, 操作模式, 外部路径, 刷新控制=0, 自动刷新间隔=0, 
                  加载限制=10, 缓存策略="智能缓存", 外部遮罩输入=None, 遮罩操作="使用外部遮罩", unique_id=None):
        
        status_info = []
        current_time = time.time()
        
        # 检查刷新条件
        needs_refresh = self._check_refresh_conditions(刷新控制, 自动刷新间隔, current_time, status_info)
        
        # 处理缓存策略
        self._handle_cache_policy(缓存策略, needs_refresh, status_info)
        
        # 根据操作模式处理
        if 操作模式 == "上传模式":
            return self._handle_upload_mode(外部路径, 缓存策略, 外部遮罩输入, 
                                          遮罩操作, status_info, needs_refresh)
        
        elif 操作模式 == "目录监控模式":
            return self._handle_monitor_mode(外部路径, 加载限制, 缓存策略, 外部遮罩输入,
                                           遮罩操作, status_info, needs_refresh, current_time)
        
        else:  # 预览模式
            return self._handle_preview_mode(图片文件, 缓存策略, 外部遮罩输入, 遮罩操作, 
                                           status_info, needs_refresh)

    def _check_refresh_conditions(self, refresh_control, auto_refresh, current_time, status_info):
        """检查刷新条件"""
        needs_refresh = False
        
        if refresh_control != self.last_refresh:
            needs_refresh = True
            self.last_refresh = refresh_control
            status_info.append("🔄 手动刷新已触发")
        
        if auto_refresh > 0 and (current_time - self.last_auto_refresh) >= auto_refresh:
            needs_refresh = True
            self.last_auto_refresh = current_time
            # 显示更友好的时间描述
            if auto_refresh < 60:
                time_desc = f"{auto_refresh}秒"
            else:
                minutes = auto_refresh // 60
                seconds = auto_refresh % 60
                time_desc = f"{minutes}分{seconds}秒" if seconds > 0 else f"{minutes}分钟"
            status_info.append(f"⏰ 自动刷新 ({time_desc})")
            
        return needs_refresh

    def _handle_cache_policy(self, cache_policy, needs_refresh, status_info):
        """处理缓存策略"""
        if cache_policy == "始终刷新":
            needs_refresh = True
            status_info.append("💾 缓存策略: 始终刷新")
        elif cache_policy == "禁用缓存":
            self.cache.clear()
            self.monitor_cache.clear()
            status_info.append("💾 缓存策略: 禁用缓存")
        else:
            status_info.append("💾 缓存策略: 智能缓存")

    def _handle_upload_mode(self, external_path, cache_policy, external_mask, 
                          mask_operation, status_info, needs_refresh):
        """处理上传模式 - 修复黑屏问题"""
        if not external_path.strip():
            return self._create_empty_output("请提供要上传的文件路径")
        
        # 验证外部路径
        is_valid, validation_msg = self._validate_external_path(external_path.strip(), "upload")
        status_info.append(validation_msg)
        
        if not is_valid:
            return self._create_empty_output(f"路径验证失败: {validation_msg}")
        
        # 上传文件
        uploaded_file = self._upload_external_image(external_path.strip())
        if not uploaded_file:
            return self._create_empty_output("文件上传失败")
        
        status_info.append(f"✅ 成功上传: {uploaded_file}")
        
        # 加载上传的图片 - 修复：使用正确的路径
        image_path = os.path.join(self.input_dir, uploaded_file)
        return self._load_and_process_image(image_path, uploaded_file, cache_policy, 
                                          external_mask, mask_operation, status_info, needs_refresh)

    def _handle_monitor_mode(self, external_path, load_limit, cache_policy, external_mask,
                           mask_operation, status_info, needs_refresh, current_time):
        """处理目录监控模式 - 改进自动刷新功能"""
        if not external_path.strip():
            return self._create_empty_output("请提供要监控的目录路径")
        
        # 验证外部路径
        is_valid, validation_msg = self._validate_external_path(external_path.strip(), "monitor")
        status_info.append(validation_msg)
        
        if not is_valid:
            return self._create_empty_output(f"路径验证失败: {validation_msg}")
        
        # 获取目录文件列表
        if needs_refresh or external_path.strip() not in self.monitor_cache:
            files = self._get_directory_files(external_path.strip(), load_limit)
            self.monitor_cache[external_path.strip()] = {
                'files': files,
                'timestamp': current_time
            }
            status_info.append(f"🔄 目录文件列表已刷新 (限制: {load_limit}个文件)")
        
        files = self.monitor_cache[external_path.strip()]['files']
        
        if not files:
            return self._create_empty_output("监控目录中没有图片文件")
        
        # 获取最新文件
        latest_file = files[0]
        cache_key = f"monitor_{latest_file}"
        
        # 检查是否需要加载新文件
        file_changed = latest_file != self.monitor_last_file
        file_updated = self._is_file_updated(latest_file, current_time)
        
        # 如果文件有变化或需要刷新，则加载新文件
        if (needs_refresh or file_changed or file_updated or 
            cache_policy == "始终刷新" or cache_key not in self.cache):
            
            # 加载外部图片
            image, mask, file_info = self._load_external_image(latest_file)
            if image is None:
                return self._create_empty_output(f"无法加载图片: {latest_file}")
            
            # 处理遮罩
            final_mask = self._process_external_mask(mask, external_mask, mask_operation)
            status_info.append(self._get_mask_status(external_mask, mask_operation))
            
            # 更新缓存
            if cache_policy != "禁用缓存":
                self.cache[cache_key] = {
                    'image': image,
                    'mask': final_mask
                }
            
            # 更新监控状态
            self.monitor_last_file = latest_file
            self.monitor_last_time = current_time
            
            status_info.append(f"✅ 已加载最新图片: {os.path.basename(latest_file)}")
            status_info.append(file_info)
            
            return (image, final_mask, "\n".join(status_info))
        else:
            # 使用缓存
            status_info.append(f"使用缓存图片: {os.path.basename(latest_file)}")
            cached_data = self.cache[cache_key]
            return (cached_data['image'], cached_data['mask'], "\n".join(status_info))

    def _is_file_updated(self, file_path, current_time):
        """检查文件是否已更新"""
        try:
            if not os.path.exists(file_path):
                return False
            
            # 获取文件修改时间
            mod_time = os.path.getmtime(file_path)
            
            # 如果文件修改时间晚于上次加载时间，说明文件已更新
            return mod_time > self.monitor_last_time
        except:
            return False

    def _handle_preview_mode(self, image, cache_policy, external_mask, mask_operation,
                           status_info, needs_refresh):
        """处理预览模式"""
        if not image:
            return self._create_empty_output("未选择图片文件")
        
        image_path = folder_paths.get_annotated_filepath(image)
        
        if not os.path.exists(image_path):
            return self._create_empty_output(f"图片文件不存在: {image}")
        
        cache_key = f"preview_{image}"
        
        if (cache_key in self.cache and not needs_refresh and cache_policy != "始终刷新"):
            status_info.append("使用缓存图片")
            cached_data = self.cache[cache_key]
            return (cached_data['image'], cached_data['mask'], "\n".join(status_info))
        
        return self._load_and_process_image(image_path, image, cache_policy, 
                                          external_mask, mask_operation, status_info, needs_refresh)

    def _load_and_process_image(self, image_path, image_name, cache_policy, external_mask, 
                              mask_operation, status_info, needs_refresh):
        """加载并处理图片"""
        try:
            # 使用改进的方法加载图片
            image, mask = self._load_image_improved(image_path)
            
            # 处理遮罩 - 修复遮罩编辑问题
            final_mask = self._process_external_mask(mask, external_mask, mask_operation)
            status_info.append(self._get_mask_status(external_mask, mask_operation))
            
            # 更新缓存
            cache_key = f"preview_{image_name}"
            if cache_policy != "禁用缓存":
                self.cache[cache_key] = {
                    'image': image,
                    'mask': final_mask
                }
            
            # 获取图片信息
            img_info = self._get_image_info(image_path)
            status_info.append(f"✅ 成功加载: {image_name}")
            status_info.append(img_info)
            
            return (image, final_mask, "\n".join(status_info))
            
        except Exception as e:
            return self._create_empty_output(f"加载图片失败: {e}")

    def _load_image_improved(self, image_path):
        """改进的图片加载方法 - 修复黑屏和遮罩问题"""
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            
            # 处理图片模式
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            
            # 转换为numpy数组
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            
            # 改进的遮罩处理 - 修复遮罩编辑问题
            mask_tensor = self._generate_mask_improved(i, image_array.shape)
            
            return (image_tensor, mask_tensor)
            
        except Exception as e:
            print(f"加载图片失败: {e}")
            # 备用方法
            try:
                return self._load_image_fallback(image_path)
            except Exception as e2:
                print(f"备用加载方法也失败: {e2}")
                raise e

    def _generate_mask_improved(self, image, image_shape):
        """改进的遮罩生成方法 - 修复遮罩编辑问题"""
        try:
            height, width = image_shape[1], image_shape[2]
            
            # 检查是否有alpha通道
            if hasattr(image, 'getchannel') and 'A' in image.getbands():
                try:
                    # 提取alpha通道
                    mask_array = np.array(image.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = torch.from_numpy(mask_array)
                    
                    # 确保mask维度正确 (H, W) -> (1, H, W)
                    if len(mask_tensor.shape) == 2:
                        mask_tensor = mask_tensor.unsqueeze(0)
                    
                    print(f"检测到Alpha通道，遮罩尺寸: {mask_tensor.shape}")
                    return mask_tensor
                    
                except Exception as e:
                    print(f"提取Alpha通道失败: {e}")
                    # 回退到全白遮罩
                    mask_tensor = torch.ones((1, height, width), dtype=torch.float32)
                    return mask_tensor
            else:
                # 没有alpha通道，创建全白遮罩 - 修复黑屏问题
                mask_tensor = torch.ones((1, height, width), dtype=torch.float32)
                return mask_tensor
                
        except Exception as e:
            print(f"生成遮罩失败: {e}")
            # 出错时返回全白遮罩
            height, width = image_shape[1], image_shape[2]
            mask_tensor = torch.ones((1, height, width), dtype=torch.float32)
            return mask_tensor

    def _load_image_fallback(self, image_path):
        """备用图片加载方法"""
        try:
            image = Image.open(image_path)
            
            # 转换为RGB模式
            if image.mode == 'RGBA':
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image.convert('RGB')
            
            # 转换为numpy数组
            image_array = np.array(rgb_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            
            # 生成遮罩
            mask_tensor = self._generate_mask_improved(image, image_array.shape)
            
            return image_tensor, mask_tensor
            
        except Exception as e:
            print(f"备用加载方法失败: {e}")
            raise e

    def _process_external_mask(self, original_mask, external_mask, mask_operation):
        """处理外部遮罩 - 修复遮罩编辑问题"""
        if external_mask is None:
            return original_mask
        
        # 确保外部遮罩维度正确
        if external_mask is not None:
            # 如果外部遮罩是3D的 (1,H,W)，保持原样
            if len(external_mask.shape) == 3:
                processed_external_mask = external_mask
            # 如果外部遮罩是2D的 (H,W)，添加批次维度
            elif len(external_mask.shape) == 2:
                processed_external_mask = external_mask.unsqueeze(0)
            else:
                # 其他情况使用原始遮罩
                processed_external_mask = original_mask
        else:
            processed_external_mask = original_mask
        
        # 根据操作模式处理遮罩
        if mask_operation == "使用外部遮罩":
            return processed_external_mask
        elif mask_operation == "覆盖外部遮罩":
            return processed_external_mask
        elif mask_operation == "忽略外部遮罩":
            return original_mask
        
        return original_mask

    def _get_mask_status(self, external_mask, mask_operation):
        """获取遮罩状态信息"""
        if external_mask is not None:
            if mask_operation == "使用外部遮罩":
                return "🎭 使用外部遮罩输入"
            elif mask_operation == "覆盖外部遮罩":
                return "🎭 覆盖为外部遮罩"
            elif mask_operation == "忽略外部遮罩":
                return "🎭 忽略外部遮罩"
        return "🎭 使用原始遮罩"

    def _validate_external_path(self, path, mode):
        """验证外部路径"""
        if not path or not path.strip():
            return False, "❌ 路径不能为空"
        
        path = path.strip()
        
        if mode == "upload":
            if not os.path.exists(path):
                return False, f"❌ 文件不存在: {path}"
            
            if not os.path.isfile(path):
                return False, f"❌ 路径不是文件: {path}"
            
            file_ext = os.path.splitext(path)[1].lower()
            if file_ext not in self.SUPPORTED_EXT:
                return False, f"❌ 不支持的图片格式: {file_ext}"
            
            return True, f"✅ 文件路径有效: {os.path.basename(path)}"
        
        elif mode == "monitor":
            if not os.path.exists(path):
                return False, f"❌ 目录不存在: {path}"
            
            if not os.path.isdir(path):
                return False, f"❌ 路径不是目录: {path}"
            
            return True, f"✅ 目录路径有效: {path}"
        
        return False, "❌ 未知的操作模式"

    def _upload_external_image(self, source_path):
        """上传外部图片到输入目录"""
        try:
            if not os.path.exists(source_path):
                return None
            
            # 验证图片文件
            try:
                with Image.open(source_path) as img:
                    img.verify()
            except Exception:
                return None
            
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
            return os.path.basename(target_path)
            
        except Exception as e:
            print(f"图片上传失败: {e}")
            return None

    def _load_external_image(self, image_path):
        """直接加载外部图片"""
        try:
            if not os.path.exists(image_path):
                return None, None, "文件不存在"
            
            image, mask = self._load_image_improved(image_path)
            img_info = self._get_image_info(image_path)
            
            return image, mask, img_info
            
        except Exception as e:
            return None, None, f"加载失败: {e}"

    def _get_directory_files(self, directory_path, limit=10):
        """获取目录中的图片文件"""
        try:
            if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
                return []
            
            files = []
            for ext in self.SUPPORTED_EXT:
                pattern = os.path.join(directory_path, f"*{ext}")
                files.extend(glob.glob(pattern))
            
            # 按修改时间排序
            files.sort(key=os.path.getmtime, reverse=True)
            
            # 应用限制
            if limit > 0 and len(files) > limit:
                files = files[:limit]
            
            return files
            
        except Exception as e:
            print(f"获取目录文件列表失败: {e}")
            return []

    def _get_image_info(self, image_path):
        """获取图片信息"""
        try:
            with Image.open(image_path) as img:
                dimensions = img.size
                mode = img.mode
                format_info = img.format
            
            file_size = os.path.getsize(image_path)
            
            info = f"尺寸: {dimensions[0]}x{dimensions[1]}\n"
            info += f"模式: {mode}\n"
            info += f"格式: {format_info}\n"
            info += f"大小: {self._format_file_size(file_size)}"
            
            return info
        except Exception as e:
            return f"获取图片信息失败: {e}"

    def _format_file_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names)-1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"

    def _create_empty_output(self, error_message):
        """创建空输出"""
        # 创建默认的黑色图像和白色mask - 修复黑屏问题
        empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        empty_mask = torch.ones((1, 512, 512), dtype=torch.float32)
        print(f"即时预览图片加载器错误: {error_message}")
        return (empty_image, empty_mask, error_message)

    @classmethod
    def IS_CHANGED(cls, 图片文件, **kwargs):
        """检查文件是否更改（官方方法）"""
        image_path = folder_paths.get_annotated_filepath(图片文件)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

# 节点注册
NODE_CLASS_MAPPINGS = {
    "InstantPreviewImageLoader": InstantPreviewImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantPreviewImageLoader": "即时预览图片加载器",
}