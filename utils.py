"""
MISLG Tools - 工具节点模块
提供内存优化、工作流验证、万能数据切换等实用工具
作者: MISLG
"""
import torch
import gc

class MemoryOptimizer:
    """内存优化器 - 清理GPU缓存和系统内存"""
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

    def optimize_memory(self, clear_cuda_cache, run_garbage_collect, enable_benchmark):
        status = []
        if clear_cuda_cache and torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated() / 1024**3
            status.append(f"GPU缓存: {before:.2f}GB -> {after:.2f}GB")
        
        if run_garbage_collect:
            collected = gc.collect()
            status.append(f"垃圾回收: {collected} 个对象")
            
        if enable_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            status.append("CUDA基准优化已启用")
            
        return (" | ".join(status) if status else "无操作",)


class WorkflowValidator:
    """工作流验证器 - 检查输入连接状态并自动修复"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "audio_input": ("AUDIO",),
                "video_input": ("VIDEO",),
                "latent_input": ("LATENT",),
            },
            "required": {
                "validate_connections": ("BOOLEAN", {"default": True}),
                "auto_fix_missing": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("AUDIO", "VIDEO", "LATENT", "STRING")
    RETURN_NAMES = ("audio", "video", "latent", "validation_report")
    FUNCTION = "validate_workflow"
    CATEGORY = "MISLG Tools/Utils"

    def validate_workflow(self, validate_connections, auto_fix_missing, audio_input=None, video_input=None, latent_input=None):
        report = ["=== 工作流验证报告 ==="]
        inputs_status = []
        
        if audio_input is not None:
            inputs_status.append(f"✅ 音频: {audio_input.shape if isinstance(audio_input, torch.Tensor) else type(audio_input)}")
        else:
            inputs_status.append("❌ 音频: 未连接")
            
        if video_input is not None:
            inputs_status.append(f"✅ 视频: {video_input.shape if isinstance(video_input, torch.Tensor) else type(video_input)}")
        else:
            inputs_status.append("❌ 视频: 未连接")
            
        if latent_input is not None:
            if isinstance(latent_input, dict) and "samples" in latent_input:
                inputs_status.append(f"✅ 潜在空间: {latent_input['samples'].shape}")
            else:
                inputs_status.append("⚠️ 潜在空间: 格式异常")
        else:
            inputs_status.append("❌ 潜在空间: 未连接")
            
        report.extend(inputs_status)
        fixed_audio, fixed_video, fixed_latent = audio_input, video_input, latent_input
        
        if auto_fix_missing:
            fix_actions = []
            if fixed_audio is None:
                fixed_audio = torch.zeros((1, 44100), dtype=torch.float32)
                fix_actions.append("音频 → 默认静音")
            if fixed_video is None:
                fixed_video = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                fix_actions.append("视频 → 默认黑色帧")
            if fixed_latent is None:
                fixed_latent = {"samples": torch.zeros([1, 4, 64, 64])}
                fix_actions.append("潜在空间 → 默认零张量")
                
            if fix_actions:
                report.append("=== 自动修复 ===")
                report.extend(fix_actions)
                
        connected_count = sum(1 for x in [audio_input, video_input, latent_input] if x is not None)
        if connected_count == 3:
            report.append("🎉 验证通过: 所有输入已连接")
        elif connected_count > 0:
            report.append(f"⚠️ 部分连接: {connected_count}/3 个输入已连接")
        else:
            report.append("❌ 验证失败: 没有输入连接")
            
        return (fixed_audio, fixed_video, fixed_latent, "\n".join(report))


class UniversalSwitch:
    """万能数据切换器 - 支持任意类型的二选一输入，自动透传原始类型"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("*",),
                "input2": ("*",),
            }
        }
    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("output", "status")
    FUNCTION = "switch_any"
    CATEGORY = "MISLG Tools/Switches"

    def switch_any(self, select_input, input1=None, input2=None):
        status = f"万能切换器: 选择 {select_input}"
        
        # 1. 优先返回指定输入
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
            
        # 2. 自动降级回退（防断连报错）
        elif input1 is not None:
            return (input1, f"{status} (自动回退到 input1)")
        elif input2 is not None:
            return (input2, f"{status} (自动回退到 input2)")
            
        # 3. 双未连接时安全返回 None
        return (None, f"{status} (无输入连接，输出 None)")


# ==========================================
# 节点注册映射 (供 __init__.py 动态读取)
# ==========================================
NODE_CLASS_MAPPINGS = {
    "MISLG MemoryOptimizer": MemoryOptimizer,
    "MISLG WorkflowValidator": WorkflowValidator,
    "MISLG UniversalSwitch": UniversalSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MISLG MemoryOptimizer": "🧹 内存优化",
    "MISLG WorkflowValidator": "✅ 工作流验证",
    "MISLG UniversalSwitch": "🔀 万能数据切换器",
}