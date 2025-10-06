"""
MISLG Tools - 工具节点模块
提供内存优化、工作流验证、数据切换等实用工具
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
        
        optimization_status = " | ".join(status) if status else "无操作"
        return (optimization_status,)

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
        
        # 输入状态检查
        inputs_status = []
        if audio_input is not None:
            if isinstance(audio_input, torch.Tensor):
                inputs_status.append(f"✅ 音频: {audio_input.shape}")
            else:
                inputs_status.append(f"✅ 音频: {type(audio_input)}")
        else:
            inputs_status.append("❌ 音频: 未连接")
        
        if video_input is not None:
            if isinstance(video_input, torch.Tensor):
                inputs_status.append(f"✅ 视频: {video_input.shape}")
            else:
                inputs_status.append(f"✅ 视频: {type(video_input)}")
        else:
            inputs_status.append("❌ 视频: 未连接")
        
        if latent_input is not None:
            if isinstance(latent_input, dict) and "samples" in latent_input:
                latent_shape = latent_input["samples"].shape
                inputs_status.append(f"✅ 潜在空间: {latent_shape}")
            else:
                inputs_status.append("⚠️ 潜在空间: 格式异常")
        else:
            inputs_status.append("❌ 潜在空间: 未连接")
        
        report.extend(inputs_status)
        
        # 自动修复
        fixed_audio = audio_input
        fixed_video = video_input
        fixed_latent = latent_input
        
        if auto_fix_missing:
            fix_actions = []
            
            if fixed_audio is None:
                # 创建默认音频张量 (1秒, 44100Hz, 单声道)
                fixed_audio = torch.zeros((1, 44100), dtype=torch.float32)
                fix_actions.append("音频 → 默认静音")
            
            if fixed_video is None:
                # 创建默认视频张量 (1帧, 64x64, 3通道)
                fixed_video = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                fix_actions.append("视频 → 默认黑色帧")
            
            if fixed_latent is None:
                fixed_latent = {"samples": torch.zeros([1, 4, 64, 64])}
                fix_actions.append("潜在空间 → 默认零张量")
            
            if fix_actions:
                report.append("=== 自动修复 ===")
                report.extend(fix_actions)
        
        # 验证总结
        connected_count = sum(1 for x in [audio_input, video_input, latent_input] if x is not None)
        total_count = 3
        
        if connected_count == total_count:
            report.append(f"🎉 验证通过: 所有 {total_count} 个输入已连接")
        elif connected_count > 0:
            report.append(f"⚠️ 部分连接: {connected_count}/{total_count} 个输入已连接")
        else:
            report.append("❌ 验证失败: 没有输入连接")
        
        validation_report = "\n".join(report)
        return (fixed_audio, fixed_video, fixed_latent, validation_report)

class AudioSwitch:
    """音频切换器 - 专门用于切换AUDIO类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("AUDIO",),
                "input2": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "switch_audio"
    CATEGORY = "MISLG Tools/Switches"

    def switch_audio(self, select_input, input1=None, input2=None):
        status = f"音频切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回默认音频
            status += " (使用默认音频)"
            default_audio = torch.zeros((1, 44100), dtype=torch.float32)
            return (default_audio, status)

class VideoSwitch:
    """视频切换器 - 专门用于切换VIDEO类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("VIDEO",),
                "input2": ("VIDEO",),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "status")
    FUNCTION = "switch_video"
    CATEGORY = "MISLG Tools/Switches"

    def switch_video(self, select_input, input1=None, input2=None):
        status = f"视频切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回默认视频
            status += " (使用默认视频)"
            default_video = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (default_video, status)

class LatentSwitch:
    """潜在空间切换器 - 专门用于切换LATENT类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("LATENT",),
                "input2": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "switch_latent"
    CATEGORY = "MISLG Tools/Switches"

    def switch_latent(self, select_input, input1=None, input2=None):
        status = f"潜在空间切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回默认潜在空间
            status += " (使用默认潜在空间)"
            default_latent = {"samples": torch.zeros([1, 4, 64, 64])}
            return (default_latent, status)

class ConditioningSwitch:
    """条件切换器 - 专门用于切换CONDITIONING类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("CONDITIONING",),
                "input2": ("CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "status")
    FUNCTION = "switch_conditioning"
    CATEGORY = "MISLG Tools/Switches"

    def switch_conditioning(self, select_input, input1=None, input2=None):
        status = f"条件切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回空列表
            status += " (使用空条件)"
            return ([], status)

class StringSwitch:
    """字符串切换器 - 专门用于切换STRING类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("STRING", {"multiline": True, "default": ""}),
                "input2": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "status")
    FUNCTION = "switch_string"
    CATEGORY = "MISLG Tools/Switches"

    def switch_string(self, select_input, input1=None, input2=None):
        status = f"字符串切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回空字符串
            status += " (使用空字符串)"
            return ("", status)

class IntSwitch:
    """整数切换器 - 专门用于切换INT类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("INT", {"default": 0, "min": 0, "max": 10000000}),
                "input2": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("value", "status")
    FUNCTION = "switch_int"
    CATEGORY = "MISLG Tools/Switches"

    def switch_int(self, select_input, input1=None, input2=None):
        status = f"整数切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回默认值
            status += " (使用默认值0)"
            return (0, status)

class FloatSwitch:
    """浮点数切换器 - 专门用于切换FLOAT类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000000.0, "step": 0.01}),
                "input2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000000.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("value", "status")
    FUNCTION = "switch_float"
    CATEGORY = "MISLG Tools/Switches"

    def switch_float(self, select_input, input1=None, input2=None):
        status = f"浮点数切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回默认值
            status += " (使用默认值0.0)"
            return (0.0, status)

class BooleanSwitch:
    """布尔值切换器 - 专门用于切换BOOLEAN类型数据"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input1", "input2"], {"default": "input1"}),
            },
            "optional": {
                "input1": ("BOOLEAN", {"default": False}),
                "input2": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("value", "status")
    FUNCTION = "switch_boolean"
    CATEGORY = "MISLG Tools/Switches"

    def switch_boolean(self, select_input, input1=None, input2=None):
        status = f"布尔值切换器: 选择 {select_input}"
        
        if select_input == "input1" and input1 is not None:
            return (input1, status)
        elif select_input == "input2" and input2 is not None:
            return (input2, status)
        
        # 如果选择的输入不存在，返回另一个输入或默认值
        if input1 is not None:
            status += " (回退到输入1)"
            return (input1, status)
        elif input2 is not None:
            status += " (回退到输入2)"
            return (input2, status)
        else:
            # 两个输入都为空，返回默认值
            status += " (使用默认值False)"
            return (False, status)

class SimpleAudioSwitch:
    """简单音频切换器 - 更简单的接口，避免输入缺失问题"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input_a", "input_b"], {"default": "input_a"}),
                "input_a": ("AUDIO",),
                "input_b": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "switch_audio_simple"
    CATEGORY = "MISLG Tools/Switches"

    def switch_audio_simple(self, select_input, input_a, input_b):
        status = f"简单音频切换器: 选择 {select_input}"
        
        if select_input == "input_a":
            return (input_a, status)
        else:
            return (input_b, status)

class SimpleVideoSwitch:
    """简单视频切换器 - 更简单的接口，避免输入缺失问题"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_input": (["input_a", "input_b"], {"default": "input_a"}),
                "input_a": ("VIDEO",),
                "input_b": ("VIDEO",),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "status")
    FUNCTION = "switch_video_simple"
    CATEGORY = "MISLG Tools/Switches"

    def switch_video_simple(self, select_input, input_a, input_b):
        status = f"简单视频切换器: 选择 {select_input}"
        
        if select_input == "input_a":
            return (input_a, status)
        else:
            return (input_b, status)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "MemoryOptimizer": MemoryOptimizer,
    "WorkflowValidator": WorkflowValidator,
    "AudioSwitch": AudioSwitch,
    "VideoSwitch": VideoSwitch,
    "LatentSwitch": LatentSwitch,
    "ConditioningSwitch": ConditioningSwitch,
    "StringSwitch": StringSwitch,
    "IntSwitch": IntSwitch,
    "FloatSwitch": FloatSwitch,
    "BooleanSwitch": BooleanSwitch,
    "SimpleAudioSwitch": SimpleAudioSwitch,
    "SimpleVideoSwitch": SimpleVideoSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MemoryOptimizer": "🧹 内存优化",
    "WorkflowValidator": "✅ 工作流验证",
    "AudioSwitch": "🎵 音频切换器",
    "VideoSwitch": "🎬 视频切换器",
    "LatentSwitch": "🎭 潜在空间切换器",
    "ConditioningSwitch": "🔗 条件切换器",
    "StringSwitch": "📝 文本切换器",
    "IntSwitch": "🔢 整数切换器",
    "FloatSwitch": "📊 浮点数切换器",
    "BooleanSwitch": "🔘 布尔值切换器",
    "SimpleAudioSwitch": "🎵 简单音频切换器",
    "SimpleVideoSwitch": "🎬 简单视频切换器",
}