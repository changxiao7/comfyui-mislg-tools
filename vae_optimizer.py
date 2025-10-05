"""
VAE 优化节点模块
优化 VAE 解码性能，确保输出能正常保存图片
"""

import torch
import gc
import numpy as np

class VAEDecoderOptimizer:
    """VAE 解码优化器 - 确保正常保存图片"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "输入的潜在空间数据，来自 KSampler 或其他生成节点"}),
                "vae": ("VAE", {"tooltip": "VAE 模型，用于解码潜在空间到图像"}),
                "use_tiled_decoding": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "启用分块解码\n\n✅ 优点：\n• 减少内存使用\n• 支持大尺寸图像解码\n\n❌ 缺点：\n• 可能稍微降低速度\n• 某些 VAE 模型不支持\n\n📌 建议：\n• 大图像(>1024px)或低显存时启用\n• 小图像可关闭以提高速度"
                }),
                "tile_size": ("INT", {
                    "default": 512, 
                    "min": 256, 
                    "max": 2048, 
                    "step": 64,
                    "tooltip": "分块解码的块大小\n\n💡 设置建议：\n• 4GB显存: 256-384\n• 6-8GB显存: 384-512\n• 8-12GB显存: 512-768\n• 12GB+显存: 768-1024\n\n⚠️ 注意：\n• 值越小内存使用越少但速度越慢\n• 值越大速度越快但需要更多显存"
                }),
                "memory_efficient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "内存效率优化\n\n🔧 功能：\n• 启用 CUDA 基准优化\n• 自动内存管理\n• 优化计算精度\n\n✅ 建议：\n• 通常保持启用\n• 如果遇到兼容性问题可关闭"
                }),
                "ensure_float32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "确保输出为 float32 格式\n\n🎯 关键功能：\n• 强制输出数据类型为 torch.float32\n• 防止因数据类型导致的保存错误\n\n⚠️ 重要：\n• 必须启用以确保能正常保存图片\n• 关闭可能导致无法保存输出"
                }),
                "normalize_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "标准化输出范围到 [0, 1]\n\n📊 功能：\n• 自动检测输入值范围\n• 将 [-1,1] 或其他范围转换到 [0,1]\n• 确保值在有效范围内\n\n💡 作用：\n• 防止图像过亮或过暗\n• 确保显示和保存正常"
                }),
                "fix_tensor_shape": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "修复张量形状\n\n🔄 功能：\n• 自动转换 BCHW → BHWC 格式\n• 确保正确的批次维度\n• 处理不常见的张量形状\n\n✅ 建议：\n• 通常保持启用\n• 如果遇到形状错误可关闭调试"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用调试输出\n\n📝 功能：\n• 在控制台打印详细处理信息\n• 显示状态信息在节点输出中\n• 帮助诊断解码问题\n\n🔧 调试：\n• 开发时保持启用\n• 生产环境可关闭减少日志"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "optimized_decode"
    CATEGORY = "MISLG Tools/VAE"
    DESCRIPTION = "优化的 VAE 解码器，确保输出兼容保存节点\n\n主要功能：\n• 内存优化解码\n• 数据类型和形状修复\n• 值范围标准化\n• 错误恢复机制"

    def optimized_decode(self, samples, vae, use_tiled_decoding, tile_size, memory_efficient,
                        ensure_float32, normalize_output, fix_tensor_shape, debug_output):
        
        status_messages = []
        
        # 初始状态信息
        if debug_output:
            status_messages.append("🚀 开始 VAE 解码优化处理")
            print(f"🔧 VAE解码优化启动: 分块={use_tiled_decoding}, 分块大小={tile_size}")
        
        try:
            # 1. 内存优化设置
            if memory_efficient:
                torch.backends.cudnn.benchmark = True
                if debug_output:
                    status_messages.append("✅ 内存优化已启用")
                    print("✅ 内存优化设置已应用")
            
            # 2. 清理 GPU 缓存
            if torch.cuda.is_available():
                before_memory = torch.cuda.memory_allocated() / 1024**3 if debug_output else 0
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if debug_output:
                    after_memory = torch.cuda.memory_allocated() / 1024**3
                    status_messages.append(f"🧹 GPU缓存已清理: {before_memory:.2f}GB → {after_memory:.2f}GB")
                    print(f"🧹 GPU缓存清理完成")
            
            # 3. 执行 VAE 解码
            with torch.no_grad():
                if use_tiled_decoding and hasattr(vae, 'decode_tiled'):
                    if debug_output:
                        status_messages.append(f"🔲 使用分块解码 (分块大小: {tile_size})")
                        print(f"🔲 开始分块解码，分块大小: {tile_size}")
                    
                    image = vae.decode_tiled(samples['samples'], tile_x=tile_size, tile_y=tile_size)
                    
                    if debug_output:
                        print(f"✅ 分块解码完成")
                else:
                    if debug_output:
                        status_messages.append("⚡ 使用标准解码")
                        print("⚡ 开始标准解码")
                    
                    image = vae.decode(samples['samples'])
                    
                    if debug_output:
                        print(f"✅ 标准解码完成")
            
            # 4. 记录解码后状态
            if debug_output:
                original_shape = image.shape
                original_dtype = image.dtype
                status_messages.append(f"📊 解码后: {original_shape}, {original_dtype}")
                print(f"📊 解码完成 - 形状: {original_shape}, 类型: {original_dtype}")
            
            # 5. 确保输出兼容性
            image = self.ensure_compatible_output(image, ensure_float32, normalize_output, fix_tensor_shape, debug_output)
            
            # 6. 解码后清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if debug_output:
                    status_messages.append("🧹 解码后缓存已清理")
                    print("🧹 解码后缓存清理完成")
            
            # 7. 最终状态报告
            if debug_output:
                final_status = f"✅ 解码成功 - 输出: {image.shape}, {image.dtype}"
                status_messages.append(final_status)
                print(final_status)
            
        except Exception as e:
            error_msg = f"❌ VAE 解码失败: {str(e)}"
            status_messages.append(error_msg)
            print(f"❌ VAE解码错误: {str(e)}")
            
            # 备用方案：创建兼容的空白图像
            image = self.create_compatible_fallback_image()
            fallback_msg = "🔄 使用备用兼容图像"
            status_messages.append(fallback_msg)
            print(fallback_msg)
        
        # 确保 status 始终有输出
        if not status_messages:
            status_messages.append("ℹ️ 解码完成（调试输出已禁用）")
        
        status = " | ".join(status_messages)
        return (image, status)

    def ensure_compatible_output(self, image, ensure_float32, normalize_output, fix_tensor_shape, debug_output):
        """确保输出与 ComfyUI 保存节点完全兼容"""
        
        if debug_output:
            print(f"🛠️ 开始输出兼容性处理")
            print(f"🛠️ 输入图像信息 - 形状: {image.shape}, 类型: {image.dtype}")
        
        # 处理特殊情况：形状为 (1, 1, H, W, C) 或类似的不常见形状
        if len(image.shape) == 5:
            if debug_output:
                print(f"🔧 检测到5维张量，尝试降维: {image.shape}")
            # 尝试降维到4维
            if image.shape[0] == 1 and image.shape[1] == 1:
                image = image.squeeze(0).squeeze(0)
            elif image.shape[0] == 1:
                image = image.squeeze(0)
        
        # 处理 uint8 数据类型 (|u1)
        if image.dtype == torch.uint8:
            if debug_output:
                print(f"🔧 检测到 uint8 数据类型，转换为 float32")
            image = image.float() / 255.0
        
        # 确保是 torch.Tensor
        if not isinstance(image, torch.Tensor):
            if debug_output:
                print(f"🔧 转换非Tensor输入为Tensor")
            image = torch.tensor(image)
        
        # 确保 float32 数据类型
        if ensure_float32 and image.dtype != torch.float32:
            original_dtype = image.dtype
            image = image.to(torch.float32)
            if debug_output:
                print(f"🔧 数据类型转换: {original_dtype} → float32")
        
        # 修复张量形状
        if fix_tensor_shape:
            original_shape = image.shape
            
            # 处理特殊形状 (1, 1, H, W, C) 或 (1, 1, H, C)
            if len(image.shape) == 4 and image.shape[1] == 1:
                # 形状为 (B, 1, H, W) 或 (B, 1, H, C)
                if image.shape[3] == 3:  # (B, 1, H, 3)
                    image = image.permute(0, 2, 1, 3)  # → (B, H, 1, 3)
                    if debug_output:
                        print(f"🔧 特殊形状处理: {original_shape} → {image.shape}")
                else:  # (B, 1, H, W)
                    image = image.squeeze(1)  # → (B, H, W)
                    if debug_output:
                        print(f"🔧 移除单通道维度: {original_shape} → {image.shape}")
            
            # 添加批次维度
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                if debug_output:
                    print(f"🔧 添加批次维度: {original_shape} → {image.shape}")
            
            # 转换 BCHW → BHWC
            elif len(image.shape) == 4 and image.shape[1] == 3:
                image = image.permute(0, 2, 3, 1)
                if debug_output:
                    print(f"🔧 格式转换 BCHW → BHWC: {original_shape} → {image.shape}")
        
        # 标准化输出范围
        if normalize_output:
            min_val = torch.min(image).item()
            max_val = torch.max(image).item()
            
            if debug_output:
                print(f"📊 值范围检测: [{min_val:.3f}, {max_val:.3f}]")
            
            if min_val < -0.1 or max_val > 1.1:
                if min_val >= -1.1 and max_val <= 1.1:
                    # [-1, 1] 范围转换到 [0, 1]
                    image = (image + 1.0) / 2.0
                    if debug_output:
                        print(f"🔧 范围转换 [-1,1] → [0,1]")
                else:
                    # 其他范围，使用 min-max 归一化
                    image_min = torch.min(image)
                    image_max = torch.max(image)
                    if (image_max - image_min) > 1e-6:
                        image = (image - image_min) / (image_max - image_min)
                        if debug_output:
                            print(f"🔧 范围归一化 → [0,1]")
            
            # 最终确保在 [0, 1] 范围内
            image = torch.clamp(image, 0.0, 1.0)
            if debug_output:
                final_min = torch.min(image).item()
                final_max = torch.max(image).item()
                print(f"✅ 最终值范围: [{final_min:.3f}, {final_max:.3f}]")
        
        # 最终形状验证
        if len(image.shape) != 4 or image.shape[-1] != 3:
            if debug_output:
                print(f"⚠️ 最终形状不标准: {image.shape}，尝试修复")
            image = self.fix_final_shape(image, debug_output)
        
        if debug_output:
            print(f"✅ 输出兼容性处理完成 - 最终形状: {image.shape}, 类型: {image.dtype}")
        
        return image

    def fix_final_shape(self, image, debug_output):
        """修复最终输出形状为标准的 (B, H, W, 3) 格式"""
        
        original_shape = image.shape
        
        # 处理 2D 图像 (H, W)
        if len(image.shape) == 2:
            image = image.unsqueeze(-1).repeat(1, 1, 3)  # (H, W) → (H, W, 3)
            image = image.unsqueeze(0)  # (H, W, 3) → (1, H, W, 3)
        
        # 处理 3D 图像
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # (H, W, 3)
                image = image.unsqueeze(0)  # → (1, H, W, 3)
            elif image.shape[0] == 3:  # (3, H, W)
                image = image.permute(1, 2, 0).unsqueeze(0)  # → (1, H, W, 3)
            else:  # (B, H, W) 或其他
                image = image.unsqueeze(-1).repeat(1, 1, 1, 3)  # → (B, H, W, 3)
        
        # 处理 4D 图像但不是标准格式
        elif len(image.shape) == 4:
            if image.shape[1] == 3:  # (B, 3, H, W)
                image = image.permute(0, 2, 3, 1)  # → (B, H, W, 3)
            elif image.shape[3] != 3:  # (B, H, W, C) 但 C != 3
                if image.shape[1] == 1 and image.shape[3] == 3:  # (B, 1, W, 3)
                    image = image.squeeze(1)  # → (B, W, 3)
                    # 可能需要进一步处理
        
        if debug_output:
            print(f"🔧 最终形状修复: {original_shape} → {image.shape}")
        
        return image

    def create_compatible_fallback_image(self):
        """创建完全兼容的备用图像"""
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

class SimpleVAEDecoder:
    """简单 VAE 解码器 - 最大兼容性"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "输入的潜在空间数据"}),
                "vae": ("VAE", {"tooltip": "VAE 模型，用于解码"}),
                "show_status": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "显示处理状态信息\n\n📝 功能：\n• 在输出中包含状态信息\n• 帮助了解解码过程\n• 不影响图像输出质量"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")
    FUNCTION = "simple_decode"
    CATEGORY = "MISLG Tools/VAE"
    DESCRIPTION = "简单的 VAE 解码器，最大兼容性\n\n特点：\n• 最简实现，不做额外处理\n• 最大兼容性，适用于大多数情况\n• 速度最快，但缺少优化功能"

    def simple_decode(self, samples, vae, show_status):
        status = ""
        
        try:
            if show_status:
                print("🚀 开始简单 VAE 解码")
                status = "开始解码..."
            
            # 直接使用 VAE 的标准解码
            image = vae.decode(samples["samples"])
            
            # 基本兼容性处理
            image = self.ensure_basic_compatibility(image)
            
            if show_status:
                status = f"✅ 解码成功 - 输出: {image.shape}, {image.dtype}"
                print(status)
                
        except Exception as e:
            error_msg = f"❌ 解码失败: {str(e)}"
            if show_status:
                status = error_msg
            print(error_msg)
            # 返回兼容的空白图像
            image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
        return (image, status)

    def ensure_basic_compatibility(self, image):
        """确保基本兼容性"""
        # 处理 uint8 数据类型
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        # 确保 float32
        if image.dtype != torch.float32:
            image = image.float()
        
        # 确保正确形状
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        elif len(image.shape) == 4 and image.shape[1] == 3:
            image = image.permute(0, 2, 3, 1)
        
        return image

class ImageDataTypeFix:
    """图像数据类型修复节点 - 专门解决数据类型错误"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "需要修复数据类型的图像"}),
                "force_float32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "强制转换为 float32 格式\n\n🎯 功能：\n• 解决 uint8 (|u1) 数据类型错误\n• 确保与保存节点兼容\n• 自动处理值范围"
                }),
                "fix_problematic_shapes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "修复问题形状\n\n🔄 处理形状：\n• (1, 1, H, W, C)\n• (B, 1, H, W)\n• 其他不标准形状\n• 转换为标准 (B, H, W, C)"
                }),
                "debug_info": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "显示修复信息\n\n📝 输出：\n• 原始形状和类型\n• 修复步骤\n• 最终结果"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "repair_report")
    FUNCTION = "fix_data_type"
    CATEGORY = "MISLG Tools/VAE"
    DESCRIPTION = "图像数据类型修复节点\n\n专门解决 'Cannot handle this data type' 错误"

    def fix_data_type(self, image, force_float32, fix_problematic_shapes, debug_info):
        report_lines = ["🔧 图像数据类型修复报告:"]
        
        original_shape = image.shape
        original_dtype = image.dtype
        report_lines.append(f"📊 原始数据: {original_shape}, {original_dtype}")
        
        if debug_info:
            print(f"🔧 开始修复图像数据类型: {original_shape}, {original_dtype}")
        
        # 修复数据类型
        if force_float32:
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
                report_lines.append("✅ uint8 → float32 (已归一化到 [0,1])")
                if debug_info:
                    print("✅ 修复: uint8 → float32")
            elif image.dtype != torch.float32:
                image = image.float()
                report_lines.append(f"✅ {original_dtype} → float32")
                if debug_info:
                    print(f"✅ 修复: {original_dtype} → float32")
            else:
                report_lines.append("✅ 已是 float32 格式")
        
        # 修复问题形状
        if fix_problematic_shapes:
            fixed_shapes = []
            
            # 处理 5D 张量
            if len(image.shape) == 5:
                if image.shape[0] == 1 and image.shape[1] == 1:
                    image = image.squeeze(0).squeeze(0)
                    fixed_shapes.append("移除双重批次维度")
                elif image.shape[0] == 1:
                    image = image.squeeze(0)
                    fixed_shapes.append("移除批次维度")
            
            # 处理 (1, 1, H, W) 形状
            if len(image.shape) == 4 and image.shape[0] == 1 and image.shape[1] == 1:
                if image.shape[3] == 3:  # (1, 1, H, 3)
                    image = image.permute(0, 2, 1, 3)  # → (1, H, 1, 3)
                    fixed_shapes.append("重新排列维度")
                else:
                    image = image.squeeze(1)  # → (1, H, W)
                    fixed_shapes.append("移除单通道维度")
            
            # 确保标准形状 (B, H, W, 3)
            if len(image.shape) == 3:
                if image.shape[2] == 3:  # (H, W, 3)
                    image = image.unsqueeze(0)  # → (1, H, W, 3)
                    fixed_shapes.append("添加批次维度")
                else:  # (B, H, W)
                    image = image.unsqueeze(-1).repeat(1, 1, 1, 3)  # → (B, H, W, 3)
                    fixed_shapes.append("添加RGB通道")
            
            elif len(image.shape) == 4 and image.shape[1] == 3:  # (B, 3, H, W)
                image = image.permute(0, 2, 3, 1)  # → (B, H, W, 3)
                fixed_shapes.append("BCHW → BHWC 转换")
            
            if fixed_shapes:
                shape_repair = " | ".join(fixed_shapes)
                report_lines.append(f"🔄 形状修复: {shape_repair}")
                if debug_info:
                    print(f"🔄 形状修复: {shape_repair}")
            else:
                report_lines.append("✅ 形状正常")
        
        # 最终验证
        final_shape = image.shape
        final_dtype = image.dtype
        
        report_lines.append(f"📊 修复后: {final_shape}, {final_dtype}")
        
        # 兼容性检查
        if len(final_shape) == 4 and final_shape[-1] == 3 and final_dtype == torch.float32:
            report_lines.append("🎉 修复成功 - 图像可以正常保存")
        else:
            report_lines.append("⚠️ 修复完成但形状可能仍需调整")
        
        repair_report = "\n".join(report_lines)
        
        if debug_info:
            print(f"✅ 数据类型修复完成: {final_shape}, {final_dtype}")
        
        return (image, repair_report)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "VAEDecoderOptimizer": VAEDecoderOptimizer,
    "SimpleVAEDecoder": SimpleVAEDecoder,
    "ImageDataTypeFix": ImageDataTypeFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecoderOptimizer": "⚡ VAE 解码优化",
    "SimpleVAEDecoder": "⚡ VAE 解码器(简单)",
    "ImageDataTypeFix": "🔧 图像数据类型修复",
}