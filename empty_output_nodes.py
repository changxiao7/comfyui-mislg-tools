"""
空输出节点模块
接收但不处理任何输入，当上级节点没有连接时提供默认输出
"""

import torch

class EmptyOutputNode:
    """空输出节点 - 接收但不处理任何输入"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image_input": ("IMAGE",),
                "latent_input": ("LATENT",),
                "mask_input": ("MASK",),
                "conditioning_input": ("CONDITIONING",),
            },
            "required": {
                "enable_passthrough": ("BOOLEAN", {"default": True}),
                "log_received_data": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "CONDITIONING", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "conditioning", "status")
    FUNCTION = "process_output"
    CATEGORY = "MISLG Tools/Output"
    DESCRIPTION = "空输出节点，接收输入但不处理，防止因未连接而报错"

    def process_output(self, enable_passthrough, log_received_data, image_input=None, latent_input=None, mask_input=None, conditioning_input=None):
        status_parts = []
        
        received_types = []
        if image_input is not None:
            received_types.append(f"图像({image_input.shape})")
        if latent_input is not None:
            received_types.append("潜在空间")
        if mask_input is not None:
            received_types.append(f"掩码({mask_input.shape})")
        if conditioning_input is not None:
            received_types.append("条件")
        
        if received_types:
            status_parts.append(f"✅ 接收到: {', '.join(received_types)}")
        else:
            status_parts.append("⚠️ 未接收到任何输入")
        
        if enable_passthrough:
            status_parts.append("直通模式: 输入直接输出")
            return (image_input, latent_input, mask_input, conditioning_input, " | ".join(status_parts))
        else:
            status_parts.append("直通禁用: 输出为空")
            return (None, None, None, None, " | ".join(status_parts))

class UniversalOutputNode:
    """通用输出节点 - 自动适应连接状态"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "any_input": (["IMAGE", "LATENT", "MASK", "CONDITIONING"],),
            },
            "required": {
                "output_type": (["image", "latent", "mask", "auto"], {"default": "auto"}),
                "fallback_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "fallback_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "mode_info")
    FUNCTION = "universal_output"
    CATEGORY = "MISLG Tools/Output"
    DESCRIPTION = "通用输出节点，自动适应连接状态"

    def universal_output(self, output_type, fallback_width, fallback_height, any_input=None):
        mode_info = f"输出类型: {output_type} | 回退尺寸: {fallback_width}x{fallback_height}"
        
        if any_input is not None:
            mode_info += " | ✅ 使用输入数据"
            
            if isinstance(any_input, torch.Tensor):
                if len(any_input.shape) == 4 and any_input.shape[-1] in [3, 4]:
                    return (any_input, None, None, f"📤 {mode_info} (传递图像)")
                elif len(any_input.shape) == 2:
                    return (None, None, any_input, f"📤 {mode_info} (传递掩码)")
            elif isinstance(any_input, dict) and "samples" in any_input:
                return (None, any_input, None, f"📤 {mode_info} (传递潜在空间)")
        
        mode_info += " | ⚠️ 使用回退数据"
        
        if output_type == "auto" or output_type == "image":
            image = torch.zeros((1, fallback_height, fallback_width, 3), dtype=torch.float32)
            return (image, None, None, f"🔄 {mode_info} (回退图像)")
        elif output_type == "latent":
            latent = torch.zeros([1, 4, fallback_height//8, fallback_width//8])
            latent_output = {"samples": latent}
            return (None, latent_output, None, f"🔄 {mode_info} (回退潜在空间)")
        elif output_type == "mask":
            mask = torch.ones((fallback_height, fallback_width), dtype=torch.float32)
            return (None, None, mask, f"🔄 {mode_info} (回退掩码)")
        else:
            image = torch.zeros((1, fallback_height, fallback_width, 3), dtype=torch.float32)
            return (image, None, None, f"🔄 {mode_info} (默认回退图像)")

# 节点注册
NODE_CLASS_MAPPINGS = {
    "EmptyOutputNode": EmptyOutputNode,
    "UniversalOutputNode": UniversalOutputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyOutputNode": "📤 空输出节点",
    "UniversalOutputNode": "📤 通用输出节点",
}