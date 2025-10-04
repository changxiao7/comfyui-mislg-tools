"""
工具节点模块
提供内存优化、工作流验证等实用工具
"""

import torch
import gc

class MemoryOptimizer:
    """内存优化器"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clear_cuda_cache": ("BOOLEAN", {"default": True}),
                "run_garbage_collect": ("BOOLEAN", {"default": True}),
                "enable_benchmark": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimization_status",)
    FUNCTION = "optimize_memory"
    CATEGORY = "MISLG Tools/Utils"
    DESCRIPTION = "优化内存使用，提高性能"

    def optimize_memory(self, clear_cuda_cache, run_garbage_collect, enable_benchmark):
        status = []
        
        if clear_cuda_cache and torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated() / 1024**3
            status.append(f"GPU缓存: {before:.2f}GB -> {after:.2f}GB")
        
        if run_garbage_collect:
            collected = gc.collect()
            status.append(f"垃圾回收: {collected} objects")
        
        if enable_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            status.append("CUDA基准优化已启用")
        
        optimization_status = " | ".join(status) if status else "无操作"
        return (optimization_status,)

class WorkflowValidator:
    """工作流验证器"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image_input": ("IMAGE",),
                "latent_input": ("LATENT",),
                "mask_input": ("MASK",),
            },
            "required": {
                "validate_connections": ("BOOLEAN", {"default": True}),
                "auto_fix_missing": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "MASK", "STRING")
    RETURN_NAMES = ("image", "latent", "mask", "validation_report")
    FUNCTION = "validate_workflow"
    CATEGORY = "MISLG Tools/Utils"
    DESCRIPTION = "验证工作流连接状态，自动修复缺失连接"

    def validate_workflow(self, validate_connections, auto_fix_missing, image_input=None, latent_input=None, mask_input=None):
        report = ["=== 工作流验证报告 ==="]
        
        inputs_status = []
        if image_input is not None:
            inputs_status.append(f"✅ 图像: {image_input.shape}")
        else:
            inputs_status.append("❌ 图像: 未连接")
        
        if latent_input is not None:
            if isinstance(latent_input, dict) and "samples" in latent_input:
                latent_shape = latent_input["samples"].shape
                inputs_status.append(f"✅ 潜在空间: {latent_shape}")
            else:
                inputs_status.append("⚠️ 潜在空间: 格式异常")
        else:
            inputs_status.append("❌ 潜在空间: 未连接")
        
        if mask_input is not None:
            inputs_status.append(f"✅ 掩码: {mask_input.shape}")
        else:
            inputs_status.append("❌ 掩码: 未连接")
        
        report.extend(inputs_status)
        
        fixed_image = image_input
        fixed_latent = latent_input
        fixed_mask = mask_input
        
        if auto_fix_missing:
            fix_actions = []
            
            if fixed_image is None:
                fixed_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                fix_actions.append("图像 → 默认黑色图像")
            
            if fixed_latent is None:
                fixed_latent = {"samples": torch.zeros([1, 4, 64, 64])}
                fix_actions.append("潜在空间 → 默认零张量")
            
            if fixed_mask is None:
                fixed_mask = torch.ones((512, 512), dtype=torch.float32)
                fix_actions.append("掩码 → 默认全白掩码")
            
            if fix_actions:
                report.append("=== 自动修复 ===")
                report.extend(fix_actions)
        
        connected_count = sum(1 for x in [image_input, latent_input, mask_input] if x is not None)
        total_count = 3
        
        if connected_count == total_count:
            report.append(f"🎉 验证通过: 所有 {total_count} 个输入已连接")
        elif connected_count > 0:
            report.append(f"⚠️ 部分连接: {connected_count}/{total_count} 个输入已连接")
        else:
            report.append("❌ 验证失败: 没有输入连接")
        
        validation_report = "\n".join(report)
        return (fixed_image, fixed_latent, fixed_mask, validation_report)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "MemoryOptimizer": MemoryOptimizer,
    "WorkflowValidator": WorkflowValidator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryOptimizer": "🧹 内存优化",
    "WorkflowValidator": "✅ 工作流验证",
}