"""
图片切换与变换节点模块
提供图片切换、混合、旋转、翻转等功能
已修复所有空格污染、语法错误、API兼容性与张量维度问题，完全兼容 ComfyUI
"""
import torch
import numpy as np
from PIL import Image

# 兼容新旧版本 Pillow (新版已弃用 Image.BILINEAR)
_RESAMPLE = getattr(Image, 'Resampling', Image).BILINEAR


class ImageSwitchManual:
    """图片二进一出手动切换节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_first": ("BOOLEAN", {"default": True, "label_on": "输出第一张图", "label_off": "输出第二张图"}),
            },
            "optional": {
                "image_A": ("IMAGE",),
                "image_B": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "status")
    FUNCTION = "switch_images"
    CATEGORY = "MISLG Tools/Image"

    def switch_images(self, select_first, image_A=None, image_B=None):
        if select_first:
            if image_A is not None:
                return (image_A, "✅ 输出图片A")
            elif image_B is not None:
                return (image_B, "⚠️ 图片A不存在，自动切换到图片B")
        else:
            if image_B is not None:
                return (image_B, "✅ 输出图片B")
            elif image_A is not None:
                return (image_A, "⚠️ 图片B不存在，自动切换到图片A")
        
        blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        return (blank, "⚠️ 无输入图片，输出空白图像")


class ImageSwitchAdvanced:
    """高级图片切换节点 - 支持自动回退"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_mode": (["A", "B", "auto"], {"default": "auto"}),
                "auto_fallback": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_A": ("IMAGE",),
                "image_B": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "status")
    FUNCTION = "advanced_switch"
    CATEGORY = "MISLG Tools/Image"

    def advanced_switch(self, switch_mode, auto_fallback=True, image_A=None, image_B=None):
        if switch_mode == "auto":
            if image_A is not None:
                return (image_A, "🔄 自动选择图片A")
            if image_B is not None:
                return (image_B, "🔄 自动选择图片B")
            blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank, "⚠️ 无可用图片，输出空白图像")

        if switch_mode == "A":
            if image_A is not None:
                return (image_A, "✅ 输出图片A")
            if auto_fallback and image_B is not None:
                return (image_B, "⚠️ 图片A缺失，回退到图片B")
        else:
            if image_B is not None:
                return (image_B, "✅ 输出图片B")
            if auto_fallback and image_A is not None:
                return (image_A, "⚠️ 图片B缺失，回退到图片A")

        blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        return (blank, "⚠️ 无可用图片，输出空白图像")


class ImageBlendSwitch:
    """图片混合切换节点 - 支持渐变融合"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_blend": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_A": ("IMAGE",),
                "image_B": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("output_image", "status")
    FUNCTION = "blend_images"
    CATEGORY = "MISLG Tools/Image"

    def blend_images(self, blend_factor, use_blend, image_A=None, image_B=None):
        if image_A is None and image_B is None:
            blank = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank, "⚠️ 无输入图片")
        if image_A is None:
            return (image_B, "✅ 仅图片B可用")
        if image_B is None:
            return (image_A, "✅ 仅图片A可用")

        if image_A.shape != image_B.shape:
            return (image_A, "⚠️ 尺寸不匹配，输出图片A")

        if use_blend:
            blended = image_A * (1.0 - blend_factor) + image_B * blend_factor
            return (blended, f"🔄 混合输出 (因子: {blend_factor:.2f})")
        else:
            target = image_A if blend_factor < 0.5 else image_B
            return (target, f"✅ 切换输出 (因子: {blend_factor:.2f})")


class MISLGImageRotate:
    """图像旋转节点 - 支持多批次与画布扩展"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "rotation_type": (["90_clockwise", "90_counterclockwise", "180_rotate", "270_clockwise", "270_counterclockwise"], {"default": "90_clockwise"}),
                "expand_canvas": (["true", "false"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("rotated_images", "status")
    FUNCTION = "rotate_images"
    CATEGORY = "MISLG Tools/Image"

    def rotate_images(self, images, rotation_type, expand_canvas):
        angle_map = {
            "90_clockwise": 90, "90_counterclockwise": -90, "180_rotate": 180,
            "270_clockwise": 270, "270_counterclockwise": -270
        }
        display_map = {
            "90_clockwise": "顺时针90度", "90_counterclockwise": "逆时针90度",
            "180_rotate": "180度", "270_clockwise": "顺时针270度", "270_counterclockwise": "逆时针270度"
        }
        
        angle = angle_map[rotation_type]
        expand = expand_canvas == "true"
        status = f"🔄 旋转: {display_map[rotation_type]} (扩展: {expand})"
        
        out_list = []
        for img in images:  # [H, W, C]
            img_np = (img.cpu().numpy() * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            rot_pil = pil_img.rotate(angle, expand=expand, resample=_RESAMPLE)
            rot_np = np.array(rot_pil).astype(np.float32) / 255.0
            rot_tensor = torch.from_numpy(rot_np)
            if rot_tensor.dim() == 2:  # 灰度图转RGB
                rot_tensor = rot_tensor.unsqueeze(-1).expand(-1, -1, 3)
            out_list.append(rot_tensor)
            
        return (torch.stack(out_list), status)


class MISLGImageFlip:
    """图像翻转节点 (PIL版)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "flip_type": (["horizontal", "vertical", "both"], {"default": "horizontal"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("flipped_images", "status")
    FUNCTION = "flip_images"
    CATEGORY = "MISLG Tools/Image"

    def flip_images(self, images, flip_type):
        display_map = {"horizontal": "水平翻转", "vertical": "垂直翻转", "both": "双向翻转"}
        status = f"🔄 翻转: {display_map[flip_type]} (PIL)"
        
        out_list = []
        for img in images:
            img_np = (img.cpu().numpy() * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            if flip_type == "horizontal":
                flipped_pil = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_type == "vertical":
                flipped_pil = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                flipped_pil = pil_img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
                
            flipped_np = np.array(flipped_pil).astype(np.float32) / 255.0
            out_list.append(torch.from_numpy(flipped_np))
            
        return (torch.stack(out_list), status)


class MISLGImageFlipTorch:
    """图像翻转节点 (Torch版) - 推荐，性能更高"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "flip_type": (["horizontal", "vertical", "both"], {"default": "horizontal"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("flipped_images", "status")
    FUNCTION = "flip_images_torch"
    CATEGORY = "MISLG Tools/Image"

    def flip_images_torch(self, images, flip_type):
        display_map = {"horizontal": "水平翻转", "vertical": "垂直翻转", "both": "双向翻转"}
        status = f"🔄 翻转: {display_map[flip_type]} (Torch)"
        
        if flip_type == "horizontal":
            flipped = images.flip(dims=[2])
        elif flip_type == "vertical":
            flipped = images.flip(dims=[1])
        else:
            flipped = images.flip(dims=[1, 2])
            
        return (flipped, status)


class MISLGImageTransform:
    """高级图像变换节点 - 整合翻转与旋转"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "operation": (["flip", "rotate"], {"default": "flip"}),
                "flip_type": (["horizontal", "vertical", "both"], {"default": "horizontal"}),
                "rotation_type": (["90_clockwise", "90_counterclockwise", "180_rotate"], {"default": "90_clockwise"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("transformed_images", "status")
    FUNCTION = "transform_images"
    CATEGORY = "MISLG Tools/Image"

    def transform_images(self, images, operation, flip_type, rotation_type):
        flip_map = {"horizontal": "水平翻转", "vertical": "垂直翻转", "both": "双向翻转"}
        rot_map = {"90_clockwise": "顺时针90度", "90_counterclockwise": "逆时针90度", "180_rotate": "180度"}
        
        out_list = []
        for img in images:
            img_np = (img.cpu().numpy() * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            if operation == "flip":
                if flip_type == "horizontal":
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip_type == "vertical":
                    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
                status_msg = f"🔄 变换: {flip_map[flip_type]}"
            else:
                angle = {"90_clockwise": 90, "90_counterclockwise": -90, "180_rotate": 180}[rotation_type]
                pil_img = pil_img.rotate(angle, resample=_RESAMPLE)
                status_msg = f"🔄 变换: {rot_map[rotation_type]}"
                
            out_np = np.array(pil_img).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_np)
            if out_tensor.dim() == 2:
                out_tensor = out_tensor.unsqueeze(-1).expand(-1, -1, 3)
            out_list.append(out_tensor)
            
        return (torch.stack(out_list), status_msg)


# 节点注册映射 (键名已严格清理，无空格)
NODE_CLASS_MAPPINGS = {
    "MISLG ImageSwitchManual": ImageSwitchManual,
    "MISLG ImageSwitchAdvanced": ImageSwitchAdvanced,
    "MISLG ImageBlendSwitch": ImageBlendSwitch,
    "MISLG ImageRotate": MISLGImageRotate,
    "MISLG ImageFlip": MISLGImageFlip,
    "MISLG ImageFlipTorch": MISLGImageFlipTorch,
    "MISLG ImageTransform": MISLGImageTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MISLG ImageSwitchManual": "🔄 图片手动切换 (MISLG)",
    "MISLG ImageSwitchAdvanced": "🔄 高级图片切换 (MISLG)",
    "MISLG ImageBlendSwitch": "🎨 图片混合切换 (MISLG)",
    "MISLG ImageRotate": "🔃 图像旋转 (MISLG)",
    "MISLG ImageFlip": "↔️ 图像翻转 PIL版 (MISLG)",
    "MISLG ImageFlipTorch": "⚡ 图像翻转 Torch版 (MISLG)",
    "MISLG ImageTransform": "🛠️ 图像变换综合 (MISLG)",
}