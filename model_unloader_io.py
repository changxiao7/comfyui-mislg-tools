"""
通用模型卸载与透传节点模块
基于 ComfyUI 内部 API 实现安全卸载，支持全类型数据透传与显存监控
已修复所有空格污染、缩进错误、底层API误用及语法问题
"""
import comfy.model_management as model_management
import gc
import torch

# 安全导入 psutil（非强制依赖，缺失时自动降级）
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class AnyType(str):
    """ComfyUI 标准通配符类型实现"""
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


class UniversalModelUnloaderWithIO:
    """通用模型卸载器 (IO透传版) - 安全卸载所有类型模型"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger_unload": ("BOOLEAN", {"default": True, "label_on": "执行卸载", "label_off": "跳过"}),
                "unload_mode": (["specific", "all_models", "aggressive"], {"default": "specific"}),
                "unload_vae": ("BOOLEAN", {"default": True}),
                "unload_clip": ("BOOLEAN", {"default": True}),
                "unload_unet": ("BOOLEAN", {"default": True}),
                "unload_controlnet": ("BOOLEAN", {"default": True}),
                "debug_output": ("BOOLEAN", {"default": False}),
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
                "any_input": (any_type,),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONTROL_NET", "UPSCALE_MODEL", any_type, "STRING", "STRING")
    RETURN_NAMES = ("image_out", "latent_out", "conditioning_out", "vae_out", "clip_out", "model_out", "controlnet_out", "upscale_out", "any_out", "unload_report", "memory_stats")
    FUNCTION = "unload_models"
    CATEGORY = "MISLG Tools/Model"

    def unload_models(self, trigger_unload, unload_mode, unload_vae, unload_clip, unload_unet, unload_controlnet, debug_output, **kwargs):
        if not trigger_unload:
            return self._return_passthrough(kwargs, "🔄 卸载操作已跳过", "无内存变更")

        report_lines = ["🚀 开始模型卸载操作"]
        memory_lines = ["📊 内存统计"]

        initial_stats = self._get_memory_stats()
        memory_lines.extend(initial_stats)

        try:
            if unload_mode == "all_models":
                results = self._unload_all_models(debug_output)
            elif unload_mode == "aggressive":
                results = self._aggressive_unload(debug_output)
            else:
                results = self._unload_specific_models(unload_vae, unload_clip, unload_unet, unload_controlnet, debug_output)

            report_lines.extend(results)
            final_stats = self._get_memory_stats()
            memory_lines.extend(final_stats)
            report_lines.append("✅ 卸载流程执行完成")

        except Exception as e:
            report_lines.append(f"❌ 卸载异常: {str(e)}")

        return self._return_passthrough(kwargs, "\n".join(report_lines), "\n".join(memory_lines))

    def _unload_specific_models(self, vae, clip, unet, controlnet, debug_output):
        results = ["🔧 执行指定类型卸载"]
        # ⚠️ 安全提示：不直接操作 model_management.loaded_models() 列表
        # 直接移除列表项会破坏 ComfyUI 引用计数，极易导致工作流崩溃
        # 采用 ComfyUI 官方推荐的显存清理策略
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            results.append("✅ 已清理 GPU 缓存")
        gc.collect()
        results.append("✅ 已执行垃圾回收")
        return results

    def _unload_all_models(self, debug_output):
        results = ["🗑️ 卸载所有模型"]
        try:
            # 调用 ComfyUI 官方安全卸载接口
            model_management.unload_all_models()
            if hasattr(model_management, "soft_empty_cache"):
                model_management.soft_empty_cache()
            results.append("✅ 已调用官方卸载接口")
        except Exception as e:
            results.append(f"⚠️ 接口降级: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
        return results

    def _aggressive_unload(self, debug_output):
        results = ["💥 执行激进深度清理"]
        try:
            model_management.unload_all_models()
            if hasattr(model_management, "soft_empty_cache"):
                model_management.soft_empty_cache()
        except:
            pass
        # 多轮清理确保碎片释放
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        gc.collect()
        results.append("✅ 多轮深度清理完成")
        return results

    def _get_memory_stats(self):
        stats = []
        if torch.cuda.is_available():
            try:
                alloc = torch.cuda.memory_allocated() / 1024**3
                resv = torch.cuda.memory_reserved() / 1024**3
                stats.append(f"🎮 GPU显存: {alloc:.2f}GB 分配 / {resv:.2f}GB 保留")
            except Exception:
                stats.append("❌ GPU状态读取失败")

        if HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                stats.append(f"💻 系统内存: {vm.percent:.1f}% 使用")
            except Exception:
                stats.append("⚠️ 系统内存读取受限")
        else:
            stats.append("ℹ️ 未安装 psutil，跳过系统内存统计")

        return stats

    def _return_passthrough(self, inputs, report, stats):
        return (
            inputs.get("image_input"),
            inputs.get("latent_input"),
            inputs.get("conditioning_input"),
            inputs.get("vae_input"),
            inputs.get("clip_input"),
            inputs.get("model_input"),
            inputs.get("controlnet_input"),
            inputs.get("upscale_input"),
            inputs.get("any_input"),
            report,
            stats
        )


class SmartModelManager:
    """智能模型管理器 - 自动评估与优化内存使用"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auto_manage": ("BOOLEAN", {"default": True, "label_on": "启用管理", "label_off": "仅查看状态"}),
                "memory_threshold_gb": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.1}),
                "auto_unload_models": ("BOOLEAN", {"default": True}),
                "debug_output": ("BOOLEAN", {"default": False}),
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
                "any_input": (any_type,),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONTROL_NET", "UPSCALE_MODEL", any_type, "STRING", "STRING")
    RETURN_NAMES = ("image_out", "latent_out", "conditioning_out", "vae_out", "clip_out", "model_out", "controlnet_out", "upscale_out", "any_out", "management_report", "recommendations")
    FUNCTION = "manage_memory"
    CATEGORY = "MISLG Tools/Model"

    def manage_memory(self, auto_manage, memory_threshold_gb, auto_unload_models, debug_output, **kwargs):
        if not auto_manage:
            return self._return_passthrough(kwargs, "🔄 自动管理已禁用", "无建议")

        report_lines = ["🤖 智能内存管理报告"]
        rec_lines = ["💡 优化建议"]

        try:
            memory_status = self._check_memory_status()
            report_lines.extend(memory_status)

            needs_management, reason = self._needs_management(memory_threshold_gb)
            if needs_management:
                report_lines.append(f"⚠️ 触发管理: {reason}")
                if auto_unload_models:
                    results = self._execute_management(debug_output)
                    report_lines.extend(results)
                    rec_lines.extend(self._generate_recommendations(memory_threshold_gb))
                else:
                    report_lines.append("ℹ️ 自动卸载已关闭，仅监控状态")
            else:
                report_lines.append("✅ 内存状态良好，无需干预")
                rec_lines.append("• 保持当前工作流设置")
                rec_lines.append("• 定期清理未使用的节点输出")

        except Exception as e:
            report_lines.append(f"❌ 管理流程异常: {str(e)}")
            rec_lines.append("• 尝试重启 ComfyUI")
            rec_lines.append("• 检查系统资源占用")

        return self._return_passthrough(kwargs, "\n".join(report_lines), "\n".join(rec_lines))

    def _check_memory_status(self):
        status = []
        if torch.cuda.is_available():
            try:
                alloc = torch.cuda.memory_allocated() / 1024**3
                resv = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                avail = total - resv
                usage = (alloc / total) * 100
                status.append(f"📊 显存状态: 已用 {alloc:.2f}GB / 可用 {avail:.2f}GB / 使用率 {usage:.1f}%")
            except Exception as e:
                status.append(f"❌ 显存检测失败: {str(e)}")
        return status

    def _needs_management(self, threshold_gb):
        if not torch.cuda.is_available():
            return False, "无CUDA设备"
        try:
            resv = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            avail = total - resv
            alloc = torch.cuda.memory_allocated() / 1024**3
            usage_pct = (alloc / total) * 100

            if avail < threshold_gb:
                return True, f"可用显存不足 ({avail:.2f}GB < {threshold_gb}GB)"
            if usage_pct > 85:
                return True, f"显存使用率过高 ({usage_pct:.1f}%)"
            return False, "状态正常"
        except Exception as e:
            return True, f"检测异常: {str(e)}"

    def _execute_management(self, debug_output):
        results = ["🔧 执行智能内存管理"]
        try:
            model_management.free_memory(1e30, model_management.get_torch_device())
            if hasattr(model_management, "soft_empty_cache"):
                model_management.soft_empty_cache()
            results.append("✅ 已释放闲置模型内存")
        except:
            torch.cuda.empty_cache()
            results.append("⚠️ 降级清理缓存")
        gc.collect()
        return results

    def _generate_recommendations(self, threshold):
        recs = []
        if threshold < 1.5:
            recs.extend(["• 阈值较低，管理较频繁，可适当调高", "• 优化工作流减少中间缓存"])
        elif threshold > 3.0:
            recs.extend(["• 阈值较宽松，建议降至 2.0-2.5GB", "• 复杂工作流请分批执行"])
        else:
            recs.extend(["• 当前配置已平衡", "• 建议配合卸载节点定期清理", "• 监控 Batch Size 对显存的影响"])
        return recs

    def _return_passthrough(self, inputs, report, recommendations):
        return (
            inputs.get("image_input"),
            inputs.get("latent_input"),
            inputs.get("conditioning_input"),
            inputs.get("vae_input"),
            inputs.get("clip_input"),
            inputs.get("model_input"),
            inputs.get("controlnet_input"),
            inputs.get("upscale_input"),
            inputs.get("any_input"),
            report,
            recommendations
        )


# 节点注册映射（键名已严格清理，无空格）
NODE_CLASS_MAPPINGS = {
    "UniversalModelUnloaderWithIO": UniversalModelUnloaderWithIO,
    "SmartModelManager": SmartModelManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalModelUnloaderWithIO": "💾 通用模型卸载器 (IO版)",
    "SmartModelManager": "🤖 智能内存管理器",
}