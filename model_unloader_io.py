"""
通用模型卸载节点 - 基于ComfyUI内部API的高效版本
完整支持所有ComfyUI数据类型
"""

import comfy.model_management as model_management
import gc
import torch
import time
import psutil
from typing import Any, Dict, List, Tuple

class AnyType(str):
    """通配符类型，用于支持任意输入类型"""
    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class UniversalModelUnloaderWithIO:
    """通用模型卸载器 - 基于ComfyUI内部API的高效版本"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_unload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "触发模型卸载操作"
                }),
                "unload_mode": (["specific", "all_models", "aggressive"], {
                    "default": "specific",
                    "tooltip": "卸载模式\n• specific: 卸载指定类型模型\n• all_models: 卸载所有模型\n• aggressive: 强制深度清理"
                }),
                # 模型类型选择开关
                "unload_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 VAE 模型"
                }),
                "unload_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 CLIP 模型"
                }),
                "unload_unet": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 UNet 模型"
                }),
                "unload_controlnet": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 ControlNet 模型"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用调试输出"
                }),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "latent_input": ("LATENT",),
                "conditioning_input": ("CONDITIONING",),
                "vae_input": ("VAE",),
                "clip_input": ("CLIP",),
                "model_input": ("MODEL",),
                "controlnet_input": ("CONTROL_NET",),
                "upscale_input": ("UPSCALE_MODEL",),
                "any_input": (any,),  # 通配符输入，支持任意类型
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONTROL_NET", "UPSCALE_MODEL", any, "STRING", "STRING")
    RETURN_NAMES = ("image_out", "latent_out", "conditioning_out", "vae_out", "clip_out", "model_out", "controlnet_out", "upscale_out", "any_out", "unload_report", "memory_stats")
    FUNCTION = "unload_models"
    CATEGORY = "MISLG Tools/Model Management"
    DESCRIPTION = "通用模型卸载器 - 基于ComfyUI内部API的高效版本"

    def unload_models(self, 
                     trigger_unload: bool = True,
                     unload_mode: str = "specific",
                     unload_vae: bool = True,
                     unload_clip: bool = True,
                     unload_unet: bool = True,
                     unload_controlnet: bool = True,
                     debug_output: bool = False,
                     **kwargs):
        
        if not trigger_unload:
            return self._return_passthrough(kwargs, "🔄 卸载操作未触发", "无操作")
        
        report_lines = ["🚀 开始模型卸载操作"]
        memory_lines = ["📊 内存统计:"]
        
        # 记录初始内存状态
        initial_stats = self.get_memory_stats()
        memory_lines.extend(initial_stats)
        
        try:
            # 执行模型卸载
            if unload_mode == "all_models":
                unload_results = self.unload_all_models(debug_output)
            elif unload_mode == "aggressive":
                unload_results = self.aggressive_unload(debug_output)
            else:  # specific mode
                unload_results = self.unload_specific_models(
                    unload_vae, unload_clip, unload_unet, unload_controlnet, 
                    kwargs, debug_output
                )
            
            report_lines.extend(unload_results)
            
            # 记录最终内存状态
            final_stats = self.get_memory_stats()
            memory_saved = self.calculate_memory_saved(initial_stats, final_stats)
            
            memory_lines.extend(final_stats)
            memory_lines.append(f"💾 总计释放: {memory_saved}")
            
            report_lines.append("✅ 模型卸载完成")
            
        except Exception as e:
            error_msg = f"❌ 模型卸载失败: {str(e)}"
            report_lines.append(error_msg)
            if debug_output:
                print(f"❌ 卸载错误: {str(e)}")
        
        return self._return_passthrough(kwargs, "\n".join(report_lines), "\n".join(memory_lines))

    def unload_specific_models(self, unload_vae, unload_clip, unload_unet, unload_controlnet, inputs, debug_output):
        """卸载指定类型的模型"""
        results = []
        models_unloaded = 0
        
        # 使用ComfyUI的内部API卸载模型
        loaded_models = model_management.loaded_models()
        
        # 卸载传入的特定模型
        if unload_vae and inputs.get("vae_input") is not None:
            vae_model = inputs.get("vae_input")
            if vae_model in loaded_models:
                loaded_models.remove(vae_model)
                models_unloaded += 1
                results.append("✅ 卸载 VAE 模型")
                if debug_output:
                    print(" - VAE模型从内存中移除")
        
        if unload_clip and inputs.get("clip_input") is not None:
            clip_model = inputs.get("clip_input")
            if clip_model in loaded_models:
                loaded_models.remove(clip_model)
                models_unloaded += 1
                results.append("✅ 卸载 CLIP 模型")
                if debug_output:
                    print(" - CLIP模型从内存中移除")
        
        if unload_unet and inputs.get("model_input") is not None:
            unet_model = inputs.get("model_input")
            if unet_model in loaded_models:
                loaded_models.remove(unet_model)
                models_unloaded += 1
                results.append("✅ 卸载 UNet 模型")
                if debug_output:
                    print(" - UNet模型从内存中移除")
        
        if unload_controlnet and inputs.get("controlnet_input") is not None:
            controlnet_model = inputs.get("controlnet_input")
            if controlnet_model in loaded_models:
                loaded_models.remove(controlnet_model)
                models_unloaded += 1
                results.append("✅ 卸载 ControlNet 模型")
                if debug_output:
                    print(" - ControlNet模型从内存中移除")
        
        # 强制释放内存
        if models_unloaded > 0:
            model_management.free_memory(1e30, model_management.get_torch_device(), loaded_models)
            model_management.soft_empty_cache(True)
            
            # 清理缓存
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                if debug_output:
                    print(" - 缓存清理完成")
            except Exception as e:
                if debug_output:
                    print(f"   - 缓存清理失败: {str(e)}")
        
        if models_unloaded == 0:
            results.append("ℹ️ 未找到可卸载的指定模型")
        
        results.append(f"📦 总计卸载: {models_unloaded} 个模型")
        
        return results

    def unload_all_models(self, debug_output):
        """卸载所有模型"""
        results = []
        
        if debug_output:
            print(" - 卸载所有模型...")
        
        # 使用ComfyUI的内部API卸载所有模型
        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        
        # 深度清理缓存
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if debug_output:
                print(" - 深度缓存清理完成")
        except Exception as e:
            if debug_output:
                print(f"   - 缓存清理失败: {str(e)}")
        
        results.append("✅ 卸载所有模型")
        results.append("🧹 执行深度缓存清理")
        
        return results

    def aggressive_unload(self, debug_output):
        """激进模式卸载 - 最大程度释放内存"""
        results = []
        
        if debug_output:
            print(" - 执行激进模式卸载...")
        
        # 多次卸载确保彻底清理
        for i in range(2):
            model_management.unload_all_models()
            model_management.soft_empty_cache(True)
            
            if debug_output:
                print(f" - 第 {i+1} 轮卸载完成")
        
        # 多次清理缓存
        for i in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        if debug_output:
            print(" - 激进模式卸载完成")
        
        results.append("💥 激进模式卸载完成")
        results.append("🔁 执行多轮深度清理")
        results.append("🧹 彻底释放GPU内存")
        
        return results

    def _return_passthrough(self, inputs, report, stats):
        """返回传递的数据和报告"""
        return (
            inputs.get("image_input"),      # IMAGE
            inputs.get("latent_input"),     # LATENT
            inputs.get("conditioning_input"), # CONDITIONING
            inputs.get("vae_input"),        # VAE
            inputs.get("clip_input"),       # CLIP
            inputs.get("model_input"),      # MODEL
            inputs.get("controlnet_input"), # CONTROL_NET
            inputs.get("upscale_input"),    # UPSCALE_MODEL
            inputs.get("any_input"),        # any
            report,                         # STRING
            stats                           # STRING
        )

    def get_memory_stats(self):
        """获取内存统计信息"""
        stats = []
        
        # GPU 内存统计
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                
                stats.append(f"🎮 GPU显存: {allocated:.2f}GB / {reserved:.2f}GB")
                
            except Exception as e:
                stats.append(f"❌ GPU统计失败: {str(e)}")
        
        # 系统内存统计
        try:
            import psutil
            virtual_memory = psutil.virtual_memory()
            process = psutil.Process()
            
            system_used = virtual_memory.used / 1024**3
            system_total = virtual_memory.total / 1024**3
            process_memory = process.memory_info().rss / 1024**3
            
            stats.append(f"💻 系统内存: {system_used:.1f}GB / {system_total:.1f}GB")
            stats.append(f"🐍 进程内存: {process_memory:.2f}GB")
            
        except ImportError:
            stats.append("ℹ️ 需要psutil库获取系统内存信息")
        except Exception as e:
            stats.append(f"❌ 系统统计失败: {str(e)}")
        
        return stats

    def calculate_memory_saved(self, initial_stats, final_stats):
        """计算释放的内存大小"""
        # 简化计算，实际值通过内存统计显示
        return "通过内存统计查看具体释放量"

class SmartModelManager:
    """智能模型管理器 - 基于ComfyUI内部API"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_manage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用自动内存管理"
                }),
                "memory_threshold_gb": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "内存警戒阈值 (GB)"
                }),
                "auto_unload_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自动卸载不活跃模型"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用调试输出"
                }),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "latent_input": ("LATENT",),
                "conditioning_input": ("CONDITIONING",),
                "vae_input": ("VAE",),
                "clip_input": ("CLIP",),
                "model_input": ("MODEL",),
                "controlnet_input": ("CONTROL_NET",),
                "upscale_input": ("UPSCALE_MODEL",),
                "any_input": (any,),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONTROL_NET", "UPSCALE_MODEL", any, "STRING", "STRING")
    RETURN_NAMES = ("image_out", "latent_out", "conditioning_out", "vae_out", "clip_out", "model_out", "controlnet_out", "upscale_out", "any_out", "management_report", "recommendations")
    FUNCTION = "manage_memory"
    CATEGORY = "MISLG Tools/Model Management"
    DESCRIPTION = "智能模型管理器 - 基于ComfyUI内部API"

    def manage_memory(self, 
                     auto_manage: bool = True,
                     memory_threshold_gb: float = 2.0,
                     auto_unload_models: bool = True,
                     debug_output: bool = False,
                     **kwargs):
        
        if not auto_manage:
            return self._return_passthrough(kwargs, "🔄 自动管理已禁用", "无建议")
        
        report_lines = ["🤖 智能内存管理报告:"]
        recommendation_lines = ["💡 优化建议:"]
        
        try:
            # 检查内存状态
            memory_status = self.check_memory_status()
            report_lines.extend(memory_status)
            
            # 检查是否需要管理
            needs_management, reason = self.needs_memory_management(memory_threshold_gb)
            
            if needs_management:
                report_lines.append(f"⚠️ 需要内存管理: {reason}")
                
                # 执行内存管理
                if auto_unload_models:
                    management_results = self.execute_auto_management(debug_output)
                    report_lines.extend(management_results)
                else:
                    report_lines.append("ℹ️ 自动卸载已禁用，仅进行监控")
                
                # 生成建议
                recommendations = self.generate_recommendations(memory_threshold_gb)
                recommendation_lines.extend(recommendations)
                
            else:
                report_lines.append("✅ 内存状态良好")
                recommendation_lines.append("• 继续保持当前设置")
                
        except Exception as e:
            report_lines.append(f"❌ 内存管理失败: {str(e)}")
            recommendation_lines.append("• 检查系统状态")
        
        return self._return_passthrough(kwargs, "\n".join(report_lines), "\n".join(recommendation_lines))

    def execute_auto_management(self, debug_output):
        """执行自动内存管理"""
        results = []
        
        if debug_output:
            print("🤖 执行自动内存管理...")
        
        # 使用ComfyUI的内部API进行内存管理
        model_management.free_memory(1e30, model_management.get_torch_device())
        model_management.soft_empty_cache(True)
        
        # 清理缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        results.append("✅ 自动内存管理完成")
        results.append("🧹 清理不活跃模型")
        
        return results

    def check_memory_status(self):
        """检查内存状态"""
        status = []
        
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                available = total - reserved
                
                status.append(f"📊 显存状态:")
                status.append(f"  • 已使用: {allocated:.2f}GB")
                status.append(f"  • 已保留: {reserved:.2f}GB") 
                status.append(f"  • 可用: {available:.2f}GB")
                status.append(f"  • 使用率: {(allocated/total)*100:.1f}%")
                
            except Exception as e:
                status.append(f"❌ 显存检查失败: {str(e)}")
        
        return status

    def needs_memory_management(self, threshold_gb):
        """检查是否需要内存管理"""
        if not torch.cuda.is_available():
            return False, "无CUDA设备"
        
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available = total - reserved
            
            if available < threshold_gb:
                return True, f"可用显存不足 ({available:.2f}GB < {threshold_gb}GB)"
            
            usage_percent = (allocated / total) * 100
            if usage_percent > 85:
                return True, f"显存使用率过高 ({usage_percent:.1f}%)"
            
            return False, "内存状态正常"
            
        except Exception as e:
            return True, f"检查失败: {str(e)}"

    def generate_recommendations(self, threshold):
        """生成优化建议"""
        recommendations = []
        
        if threshold < 1.5:
            recommendations.extend([
                "• 考虑提高内存阈值以减少频繁管理",
                "• 优化工作流减少内存使用",
                "• 使用更小的模型尺寸"
            ])
        elif threshold > 3.0:
            recommendations.extend([
                "• 当前阈值设置较为宽松",
                "• 可适当降低阈值以更积极管理",
                "• 监控内存使用模式"
            ])
        else:
            recommendations.extend([
                "• 当前设置平衡良好",
                "• 继续监控内存使用情况",
                "• 根据需要调整阈值"
            ])
        
        return recommendations

    def _return_passthrough(self, inputs, report, recommendations):
        """返回传递的数据和报告"""
        return (
            inputs.get("image_input"),      # IMAGE
            inputs.get("latent_input"),     # LATENT
            inputs.get("conditioning_input"), # CONDITIONING
            inputs.get("vae_input"),        # VAE
            inputs.get("clip_input"),       # CLIP
            inputs.get("model_input"),      # MODEL
            inputs.get("controlnet_input"), # CONTROL_NET
            inputs.get("upscale_input"),    # UPSCALE_MODEL
            inputs.get("any_input"),        # any
            report,                         # STRING
            recommendations                 # STRING
        )

# 节点注册
NODE_CLASS_MAPPINGS = {
    "UniversalModelUnloaderWithIO": UniversalModelUnloaderWithIO,
    "SmartModelManager": SmartModelManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalModelUnloaderWithIO": "💾 通用模型卸载器 (高效版)",
    "SmartModelManager": "🤖 智能模型管理器",
}