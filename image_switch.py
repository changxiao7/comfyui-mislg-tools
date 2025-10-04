"""
图片切换节点模块
提供图片二进一出手动切换功能，支持单个输入
"""

import torch

class ImageSwitchManual:
    """
    图片二进一出手动切换节点
    支持两个图片输入，通过按钮手动切换输出
    单个输入也可以正常工作
    """
    
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
    DESCRIPTION = "手动切换两个输入图片的输出"

    def switch_images(self, select_first, image_A=None, image_B=None):
        status = ""
        
        # 如果选择第一张图且第一张图存在
        if select_first and image_A is not None:
            status = "✅ 输出图片A"
            return (image_A, status)
        
        # 如果选择第二张图且第二张图存在
        if not select_first and image_B is not None:
            status = "✅ 输出图片B"
            return (image_B, status)
        
        # 如果选择的图片不存在，尝试返回另一张图
        if select_first and image_A is None and image_B is not None:
            status = "⚠️ 第一张图不存在，自动切换到第二张图"
            return (image_B, status)
        
        if not select_first and image_B is None and image_A is not None:
            status = "⚠️ 第二张图不存在，自动切换到第一张图"
            return (image_A, status)
        
        # 如果两张图都不存在，创建一张空白图片
        status = "⚠️ 没有输入图片，创建空白图片"
        blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        return (blank_image, status)

class ImageSwitchAdvanced:
    """
    高级图片切换节点 - 带有更多控制选项
    """
    
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
    DESCRIPTION = "高级图片切换，支持回退图片和状态反馈"

    def advanced_switch(self, switch_mode, auto_fallback=True, image_A=None, image_B=None):
        status = ""
        
        # 自动模式：选择第一个可用的图像
        if switch_mode == "auto":
            if image_A is not None:
                status = "🔄 自动选择图片A"
                return (image_A, status)
            elif image_B is not None:
                status = "🔄 自动选择图片B"
                return (image_B, status)
            else:
                status = "⚠️ 没有可用图片，创建空白图片"
                blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (blank_image, status)
        
        # 根据模式选择图片
        if switch_mode == "A":
            if image_A is not None:
                status = "✅ 输出图片A"
                return (image_A, status)
            elif auto_fallback and image_B is not None:
                status = "⚠️ 图片A不存在，自动回退到图片B"
                return (image_B, status)
        else:  # switch_mode == "B"
            if image_B is not None:
                status = "✅ 输出图片B"
                return (image_B, status)
            elif auto_fallback and image_A is not None:
                status = "⚠️ 图片B不存在，自动回退到图片A"
                return (image_A, status)
        
        # 如果都没有图片，创建空白图片
        status = "⚠️ 没有可用图片，创建空白图片"
        blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        return (blank_image, status)

class ImageBlendSwitch:
    """
    图片混合切换节点 - 支持渐变切换
    """
    
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
    DESCRIPTION = "图片混合切换，支持渐变效果"

    def blend_images(self, blend_factor, use_blend, image_A=None, image_B=None):
        status = ""
        
        # 检查输入
        if image_A is None and image_B is None:
            status = "⚠️ 没有输入图片，创建空白图片"
            blank_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (blank_image, status)
        
        if image_A is None:
            status = "✅ 只有图片B可用"
            return (image_B, status)
        
        if image_B is None:
            status = "✅ 只有图片A可用"
            return (image_A, status)
        
        # 检查图像尺寸是否匹配
        if image_A.shape != image_B.shape:
            status = "⚠️ 图像尺寸不匹配，使用图片A"
            return (image_A, status)
        
        # 混合图像
        if use_blend:
            blended_image = image_A * (1.0 - blend_factor) + image_B * blend_factor
            status = f"🔄 混合图像 (混合因子: {blend_factor:.2f})"
            return (blended_image, status)
        else:
            # 根据混合因子选择图像
            if blend_factor < 0.5:
                status = f"✅ 选择图片A (混合因子: {blend_factor:.2f})"
                return (image_A, status)
            else:
                status = f"✅ 选择图片B (混合因子: {blend_factor:.2f})"
                return (image_B, status)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageSwitchManual": ImageSwitchManual,
    "ImageSwitchAdvanced": ImageSwitchAdvanced,
    "ImageBlendSwitch": ImageBlendSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSwitchManual": "🔄 图片手动切换",
    "ImageSwitchAdvanced": "🔄 高级图片切换",
    "ImageBlendSwitch": "🔄 图片混合切换",
}