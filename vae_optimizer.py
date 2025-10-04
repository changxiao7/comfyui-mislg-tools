"""
VAE 优化节点模块
优化 VAE 解码性能，确保输出能正常保存图片
"""

import torch
import gc

class VAEDecoderOptimizer:
    """VAE 解码优化器 - 确保正常保存图片"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "use_tiled_decoding": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "memory_efficient": ("BOOLEAN", {"default": True}),
                "ensure_float32": ("BOOLEAN", {"default": True}),
                "normalize_output": ("BOOLEAN", {"default": True}),
                "fix_tensor_shape": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "optimized_decode"
    CATEGORY = "MISLG Tools/VAE"
    DESCRIPTION = "优化的 VAE 解码器，确保输出兼容保存节点"

    def optimized_decode(self, samples, vae, use_tiled_decoding, tile_size, memory_efficient,
                        ensure_float32, normalize_output, fix_tensor_shape):
        
        status_messages = []
        
        try:
            if memory_efficient:
                torch.backends.cudnn.benchmark = True
                status_messages.append("内存优化已启用")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                status_messages.append("GPU缓存已清理")
            
            with torch.no_grad():
                if use_tiled_decoding and hasattr(vae, 'decode_tiled'):
                    image = vae.decode_tiled(samples['samples'], tile_x=tile_size, tile_y=tile_size)
                    status_messages.append(f"使用分块解码 (分块大小: {tile_size})")
                else:
                    image = vae.decode(samples['samples'])
                    status_messages.append("使用标准解码")
            
            image = self.ensure_compatible_output(image, ensure_float32, normalize_output, fix_tensor_shape)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                status_messages.append("解码后缓存已清理")
            
            status_messages.append(f"✅ 解码成功 - 输出: {image.shape}, {image.dtype}")
            
        except Exception as e:
            error_msg = f"❌ VAE 解码失败: {str(e)}"
            status_messages.append(error_msg)
            image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            status_messages.append("🔄 使用备用兼容图像")
        
        status = " | ".join(status_messages)
        return (image, status)

    def ensure_compatible_output(self, image, ensure_float32, normalize_output, fix_tensor_shape):
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        
        if ensure_float32 and image.dtype != torch.float32:
            image = image.to(torch.float32)
        
        if fix_tensor_shape:
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            elif len(image.shape) == 4:
                if image.shape[1] == 3:
                    image = image.permute(0, 2, 3, 1)
        
        if normalize_output:
            min_val = torch.min(image)
            max_val = torch.max(image)
            
            if min_val < -0.1 or max_val > 1.1:
                if min_val >= -1.1 and max_val <= 1.1:
                    image = (image + 1.0) / 2.0
                else:
                    image_min = torch.min(image)
                    image_max = torch.max(image)
                    if (image_max - image_min) > 1e-6:
                        image = (image - image_min) / (image_max - image_min)
            
            image = torch.clamp(image, 0.0, 1.0)
        
        return image

class SimpleVAEDecoder:
    """简单 VAE 解码器 - 最大兼容性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "simple_decode"
    CATEGORY = "MISLG Tools/VAE"
    DESCRIPTION = "简单的 VAE 解码器，最大兼容性"

    def simple_decode(self, samples, vae):
        return (vae.decode(samples["samples"]),)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "VAEDecoderOptimizer": VAEDecoderOptimizer,
    "SimpleVAEDecoder": SimpleVAEDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecoderOptimizer": "⚡ VAE 解码优化",
    "SimpleVAEDecoder": "⚡ VAE 解码器(简单)",
}