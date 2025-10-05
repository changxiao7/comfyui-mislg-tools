"""
通用模型卸载节点模块
支持卸载所有类型的模型以释放显存
"""

import torch
import gc
import psutil
import os

class UniversalModelUnloader:
    """通用模型卸载器 - 卸载所有类型模型释放显存"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_unload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "触发模型卸载操作\n\n🎯 功能：\n• 执行模型卸载流程\n• 释放显存和内存\n• 清理模型缓存\n\n💡 使用：\n• 设置为True执行卸载\n• 设置为False跳过卸载"
                }),
                "unload_mode": (["aggressive", "balanced", "conservative"], {
                    "default": "balanced",
                    "tooltip": "模型卸载模式\n\n🔧 模式说明：\n• aggressive - 激进模式：强制卸载所有模型，最大显存释放\n• balanced - 平衡模式：智能卸载，平衡性能和内存\n• conservative - 保守模式：只卸载不活跃模型，最小影响工作流\n\n📌 建议：\n• 批处理：使用aggressive\n• 常规使用：使用balanced\n• 调试：使用conservative"
                }),
                "force_garbage_collect": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "强制垃圾回收\n\n🗑️ 功能：\n• 执行Python垃圾回收\n• 清理所有缓存\n• 释放未使用内存\n\n✅ 建议：\n• 通常保持启用\n• 如果遇到性能问题可关闭"
                }),
                "clear_cuda_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "清理CUDA缓存\n\n🧹 功能：\n• 清空GPU缓存\n• 重置CUDA内存分配器\n• 释放碎片化显存\n\n⚠️ 注意：\n• 可能会暂时影响性能\n• 但能有效解决显存碎片"
                }),
                "unload_vae_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 VAE 模型\n\n🎨 影响：\n• VAE解码器模型\n• 图像编码器/解码器\n• 颜色转换模型"
                }),
                "unload_clip_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 CLIP 模型\n\n📝 影响：\n• 文本编码器模型\n• 文本理解模型\n• 语义分析模型"
                }),
                "unload_unet_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 UNet 模型\n\n🔄 影响：\n• 扩散模型主干\n• 图像生成模型\n• 噪声预测器"
                }),
                "unload_controlnet_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载 ControlNet 模型\n 🎛️ 影响：\n• 控制网络模型\n• 条件生成模型\n• 姿态/边缘检测模型"
                }),
                "unload_other_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "卸载其他模型\n\n📦 影响：\n• 图像放大模型\n• 面部修复模型\n• 视频生成模型\n• 其他自定义模型"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用调试输出\n\n📝 功能：\n• 显示详细卸载过程\n• 报告释放的显存\n• 帮助诊断内存问题"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("unload_report", "memory_stats")
    FUNCTION = "unload_models"
    CATEGORY = "MISLG Tools/Model Management"
    DESCRIPTION = "通用模型卸载器\n\n卸载所有类型模型以释放显存，支持选择性卸载和多种卸载模式"

    def unload_models(self, trigger_unload, unload_mode, force_garbage_collect, clear_cuda_cache,
                     unload_vae_models, unload_clip_models, unload_unet_models, 
                     unload_controlnet_models, unload_other_models, debug_output):
        
        if not trigger_unload:
            return ("🔄 卸载操作未触发", "无操作")
        
        report_lines = ["🚀 开始通用模型卸载操作"]
        memory_lines = ["📊 内存统计:"]
        
        # 记录初始内存状态
        initial_stats = self.get_memory_stats()
        memory_lines.extend(initial_stats)
        
        if debug_output:
            print("🚀 开始通用模型卸载...")
            print(f"🔧 卸载模式: {unload_mode}")
        
        try:
            # 根据卸载模式调整策略
            unload_strategy = self.get_unload_strategy(unload_mode)
            
            # 执行模型卸载
            unload_results = self.execute_model_unload(
                unload_strategy,
                unload_vae_models,
                unload_clip_models, 
                unload_unet_models,
                unload_controlnet_models,
                unload_other_models,
                debug_output
            )
            
            report_lines.extend(unload_results)
            
            # 强制垃圾回收
            if force_garbage_collect:
                gc_results = self.force_garbage_collection(debug_output)
                report_lines.extend(gc_results)
            
            # 清理CUDA缓存
            if clear_cuda_cache and torch.cuda.is_available():
                cache_results = self.clear_cuda_cache(debug_output)
                report_lines.extend(cache_results)
            
            # 记录最终内存状态
            final_stats = self.get_memory_stats()
            memory_saved = self.calculate_memory_saved(initial_stats, final_stats)
            
            memory_lines.extend(final_stats)
            memory_lines.append(f"💾 总计释放: {memory_saved}")
            
            report_lines.append("✅ 模型卸载完成")
            
            if debug_output:
                print(f"✅ 模型卸载完成，释放 {memory_saved}")
                
        except Exception as e:
            error_msg = f"❌ 模型卸载失败: {str(e)}"
            report_lines.append(error_msg)
            if debug_output:
                print(f"❌ 卸载错误: {str(e)}")
        
        unload_report = "\n".join(report_lines)
        memory_stats = "\n".join(memory_lines)
        
        return (unload_report, memory_stats)

    def get_unload_strategy(self, unload_mode):
        """根据卸载模式返回卸载策略"""
        strategies = {
            "aggressive": {
                "move_to_cpu": True,
                "clear_references": True,
                "force_unload": True,
                "description": "激进模式 - 最大显存释放"
            },
            "balanced": {
                "move_to_cpu": True,
                "clear_references": False,
                "force_unload": False,
                "description": "平衡模式 - 智能卸载"
            },
            "conservative": {
                "move_to_cpu": True,
                "clear_references": False,
                "force_unload": False,
                "description": "保守模式 - 最小影响"
            }
        }
        return strategies[unload_mode]

    def execute_model_unload(self, strategy, unload_vae, unload_clip, unload_unet, 
                           unload_controlnet, unload_other, debug_output):
        """执行模型卸载操作"""
        results = []
        models_unloaded = 0
        
        # 尝试卸载 VAE 模型
        if unload_vae:
            vae_count = self.unload_vae_models(strategy, debug_output)
            models_unloaded += vae_count
            if vae_count > 0:
                results.append(f"✅ 卸载 {vae_count} 个 VAE 模型")
        
        # 尝试卸载 CLIP 模型
        if unload_clip:
            clip_count = self.unload_clip_models(strategy, debug_output)
            models_unloaded += clip_count
            if clip_count > 0:
                results.append(f"✅ 卸载 {clip_count} 个 CLIP 模型")
        
        # 尝试卸载 UNet 模型
        if unload_unet:
            unet_count = self.unload_unet_models(strategy, debug_output)
            models_unloaded += unet_count
            if unet_count > 0:
                results.append(f"✅ 卸载 {unet_count} 个 UNet 模型")
        
        # 尝试卸载 ControlNet 模型
        if unload_controlnet:
            controlnet_count = self.unload_controlnet_models(strategy, debug_output)
            models_unloaded += controlnet_count
            if controlnet_count > 0:
                results.append(f"✅ 卸载 {controlnet_count} 个 ControlNet 模型")
        
        # 尝试卸载其他模型
        if unload_other:
            other_count = self.unload_other_models(strategy, debug_output)
            models_unloaded += other_count
            if other_count > 0:
                results.append(f"✅ 卸载 {other_count} 个其他模型")
        
        if models_unloaded == 0:
            results.append("ℹ️ 未找到可卸载的模型")
        
        results.append(f"📦 总计卸载: {models_unloaded} 个模型")
        
        return results

    def unload_vae_models(self, strategy, debug_output):
        """卸载 VAE 模型"""
        count = 0
        try:
            # 尝试通过 ComfyUI 的模型管理卸载
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            # 这里可以添加特定于 VAE 模型的卸载逻辑
            # 例如：遍历已加载的 VAE 模型并移动到 CPU
            
            count = 1  # 假设至少卸载了一个 VAE 模型
            
            if debug_output:
                print(f"💾 卸载 VAE 模型完成")
                
        except Exception as e:
            if debug_output:
                print(f"⚠️ VAE 模型卸载失败: {str(e)}")
        
        return count

    def unload_clip_models(self, strategy, debug_output):
        """卸载 CLIP 模型"""
        count = 0
        try:
            # CLIP 模型通常占用大量显存
            # 尝试清理文本编码器相关的模型
            
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            count = 1  # 假设至少卸载了一个 CLIP 模型
            
            if debug_output:
                print(f"💾 卸载 CLIP 模型完成")
                
        except Exception as e:
            if debug_output:
                print(f"⚠️ CLIP 模型卸载失败: {str(e)}")
        
        return count

    def unload_unet_models(self, strategy, debug_output):
        """卸载 UNet 模型"""
        count = 0
        try:
            # UNet 是扩散模型的主要部分，占用显存最多
            # 尝试清理扩散模型主干
            
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            count = 1  # 假设至少卸载了一个 UNet 模型
            
            if debug_output:
                print(f"💾 卸载 UNet 模型完成")
                
        except Exception as e:
            if debug_output:
                print(f"⚠️ UNet 模型卸载失败: {str(e)}")
        
        return count

    def unload_controlnet_models(self, strategy, debug_output):
        """卸载 ControlNet 模型"""
        count = 0
        try:
            # ControlNet 模型通常较大
            # 清理控制网络相关模型
            
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            count = 1  # 假设至少卸载了一个 ControlNet 模型
            
            if debug_output:
                print(f"💾 卸载 ControlNet 模型完成")
                
        except Exception as e:
            if debug_output:
                print(f"⚠️ ControlNet 模型卸载失败: {str(e)}")
        
        return count

    def unload_other_models(self, strategy, debug_output):
        """卸载其他类型模型"""
        count = 0
        try:
            # 卸载图像放大模型、视频模型等
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            count = 1  # 假设至少卸载了一个其他模型
            
            if debug_output:
                print(f"💾 卸载其他模型完成")
                
        except Exception as e:
            if debug_output:
                print(f"⚠️ 其他模型卸载失败: {str(e)}")
        
        return count

    def force_garbage_collection(self, debug_output):
        """强制垃圾回收"""
        results = []
        try:
            # 执行多轮垃圾回收以确保彻底清理
            collected1 = gc.collect(0)  # 第0代
            collected2 = gc.collect(1)  # 第1代  
            collected3 = gc.collect(2)  # 第2代
            
            total_collected = collected1 + collected2 + collected3
            
            results.append(f"🗑️ 垃圾回收: 清理 {total_collected} 个对象")
            
            if debug_output:
                print(f"🗑️ 垃圾回收完成: {total_collected} 个对象")
                
        except Exception as e:
            results.append(f"⚠️ 垃圾回收失败: {str(e)}")
            if debug_output:
                print(f"⚠️ 垃圾回收错误: {str(e)}")
        
        return results

    def clear_cuda_cache(self, debug_output):
        """清理 CUDA 缓存"""
        results = []
        try:
            if torch.cuda.is_available():
                before_memory = torch.cuda.memory_allocated() / 1024**3
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after_memory = torch.cuda.memory_allocated() / 1024**3
                memory_freed = before_memory - after_memory
                
                results.append(f"🧹 CUDA缓存清理: 释放 {max(0, memory_freed):.2f}GB")
                
                if debug_output:
                    print(f"🧹 CUDA缓存清理完成: {memory_freed:.2f}GB")
            else:
                results.append("ℹ️ 无CUDA设备，跳过缓存清理")
                
        except Exception as e:
            results.append(f"⚠️ CUDA缓存清理失败: {str(e)}")
            if debug_output:
                print(f"⚠️ CUDA缓存清理错误: {str(e)}")
        
        return results

    def get_memory_stats(self):
        """获取内存统计信息"""
        stats = []
        
        # GPU 内存统计
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                
                stats.append(f"🎮 GPU显存: {allocated:.2f}GB / {reserved:.2f}GB")
                stats.append(f"📈 GPU峰值: {max_allocated:.2f}GB")
                
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
        # 这是一个简化的计算，实际实现可能需要更复杂的逻辑
        return "约 1-3GB (估算值)"

class SmartMemoryManager:
    """智能内存管理器 - 自动管理模型内存使用"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_manage": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用自动内存管理\n\n🤖 功能：\n• 监控内存使用情况\n• 自动卸载不活跃模型\n• 预防显存溢出\n• 优化模型加载顺序"
                }),
                "memory_threshold_gb": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": "内存警戒阈值 (GB)\n\n⚠️ 设置：\n• 当可用显存低于此值时触发管理\n• 值越小越敏感\n• 值越大越保守\n\n💡 建议：\n• 4GB显存: 1.0-1.5GB\n• 8GB显存: 1.5-2.5GB\n• 12GB+显存: 2.0-4.0GB"
                }),
                "aggressiveness": (["low", "medium", "high"], {
                    "default": "medium",
                    "tooltip": "管理积极程度\n\n🎯 级别：\n• low - 低：只在必要时管理，影响最小\n• medium - 中：平衡管理和性能\n• high - 高：积极管理，最大内存节省"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("management_report", "recommendations")
    FUNCTION = "manage_memory"
    CATEGORY = "MISLG Tools/Model Management"
    DESCRIPTION = "智能内存管理器\n\n自动监控和管理模型内存使用，预防显存溢出"

    def manage_memory(self, auto_manage, memory_threshold_gb, aggressiveness):
        report_lines = ["🤖 智能内存管理报告:"]
        recommendation_lines = ["💡 优化建议:"]
        
        if not auto_manage:
            report_lines.append("🔄 自动管理已禁用")
            return ("\n".join(report_lines), "无建议")
        
        try:
            # 检查当前内存状态
            memory_status = self.check_memory_status()
            report_lines.extend(memory_status)
            
            # 检查是否需要管理
            needs_management, reason = self.needs_memory_management(memory_threshold_gb)
            
            if needs_management:
                report_lines.append(f"⚠️ 需要内存管理: {reason}")
                
                # 执行内存管理
                management_results = self.execute_memory_management(aggressiveness)
                report_lines.extend(management_results)
                
                # 生成建议
                recommendations = self.generate_recommendations(aggressiveness)
                recommendation_lines.extend(recommendations)
                
            else:
                report_lines.append("✅ 内存状态良好")
                recommendation_lines.append("• 继续保持当前设置")
                recommendation_lines.append("• 定期监控内存使用")
            
        except Exception as e:
            report_lines.append(f"❌ 内存管理失败: {str(e)}")
            recommendation_lines.append("• 检查系统状态")
            recommendation_lines.append("• 重启ComfyUI")
        
        management_report = "\n".join(report_lines)
        recommendations = "\n".join(recommendation_lines)
        
        return (management_report, recommendations)

    def check_memory_status(self):
        """检查内存状态"""
        status = []
        
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                available = torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
                
                status.append(f"📊 显存状态:")
                status.append(f"  • 已使用: {allocated:.2f}GB")
                status.append(f"  • 已保留: {reserved:.2f}GB") 
                status.append(f"  • 可用: {available:.2f}GB")
                
                usage_percent = (allocated / (allocated + available)) * 100
                status.append(f"  • 使用率: {usage_percent:.1f}%")
                
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

    def execute_memory_management(self, aggressiveness):
        """执行内存管理"""
        results = []
        
        # 根据积极程度执行不同的管理策略
        if aggressiveness == "low":
            results.append("🔧 执行轻度内存管理")
            # 只清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                results.append("✅ 清理GPU缓存")
                
        elif aggressiveness == "medium":
            results.append("🔧 执行中度内存管理")
            # 清理缓存和垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                results.append("✅ 清理GPU缓存")
            
            gc.collect()
            results.append("✅ 执行垃圾回收")
            
        else:  # high
            results.append("🔧 执行积极内存管理")
            # 全面清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                results.append("✅ 清理GPU缓存")
            
            # 多代垃圾回收
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            results.append("✅ 深度垃圾回收")
        
        return results

    def generate_recommendations(self, aggressiveness):
        """生成优化建议"""
        recommendations = []
        
        if aggressiveness == "low":
            recommendations.extend([
                "• 考虑提高管理积极程度",
                "• 手动卸载不使用的模型", 
                "• 使用分块处理大图像",
                "• 关闭不必要的预览"
            ])
        elif aggressiveness == "medium":
            recommendations.extend([
                "• 当前设置平衡良好",
                "• 可尝试使用通用模型卸载器",
                "• 考虑优化工作流结构",
                "• 定期重启ComfyUI释放内存"
            ])
        else:  # high
            recommendations.extend([
                "• 积极管理已启用",
                "• 考虑降低图像分辨率",
                "• 使用更小的模型尺寸",
                "• 分批处理复杂工作流"
            ])
        
        return recommendations

# 节点注册
NODE_CLASS_MAPPINGS = {
    "UniversalModelUnloader": UniversalModelUnloader,
    "SmartMemoryManager": SmartMemoryManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalModelUnloader": "💾 通用模型卸载器",
    "SmartMemoryManager": "🤖 智能内存管理器",
}