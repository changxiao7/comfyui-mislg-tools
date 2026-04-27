"""
KSampler with Info Output
100% 对齐官方执行链路，仅增加信息字符串输出
兼容所有架构：SD1.5/SDXL/SD3/Flux/z-image/Lumina/RES4LYF等
"""
import comfy.samplers
import comfy.sample
import latent_preview
import torch
import comfy.utils


class KSamplerWithInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale."}),
                # 🔑 直接引用官方动态列表，自动同步 RES4LYF/社区采样器
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling.", "default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed.", "default": "normal"}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied."}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "官方 KSampler 逻辑 + 参数信息输出。完美支持 RES4LYF 及所有自定义采样器。"
    SEARCH_ALIASES = ["ksampler", "sampler", "sample", "generate", "denoise"]

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        # 官方标准流程：复制字典 → 修复通道 → 准备噪声 → 调用底层
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_samples,
            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed
        )

        out = latent.copy()
        out["samples"] = samples

        info = f"seed:{seed} | steps:{steps} | cfg:{cfg} | sampler:{sampler_name} | scheduler:{scheduler} | denoise:{denoise}"
        return (out, info)


class KSamplerAdvancedWithInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "add_noise": (["enable", "disable"], {"default": "enable", "advanced": True, "tooltip": "Enable or disable initial noise addition."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling.", "default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed.", "default": "normal"}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "advanced": True, "tooltip": "Step to start sampling from."}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "advanced": True, "tooltip": "Step to end sampling at."}),
                "return_with_leftover_noise": (["disable", "enable"], {"default": "disable", "advanced": True, "tooltip": "Return with leftover noise for chaining."}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "官方高级 KSampler 逻辑 + 参数信息输出。支持分步采样与 RES4LYF 采样器。"
    SEARCH_ALIASES = ["ksampler advanced", "advanced sampler", "sample advanced"]

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise):
        
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, noise_seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        force_full_denoise = (return_with_leftover_noise == "enable")
        disable_noise = (add_noise == "disable")

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_samples,
            denoise=1.0, disable_noise=disable_noise, start_step=start_at_step,
            last_step=end_at_step, force_full_denoise=force_full_denoise,
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed
        )

        out = latent.copy()
        out["samples"] = samples

        actual_steps = max(0, min(steps, end_at_step) - start_at_step) if start_at_step < end_at_step else 0
        info = f"seed:{noise_seed} | steps:{actual_steps} | cfg:{cfg} | sampler:{sampler_name} | scheduler:{scheduler} | range:{start_at_step}->{end_at_step}"
        return (out, info)


NODE_CLASS_MAPPINGS = {
    "KSamplerWithInfo": KSamplerWithInfo,
    "KSamplerAdvancedWithInfo": KSamplerAdvancedWithInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerWithInfo": "K采样器 (含采样信息)",
    "KSamplerAdvancedWithInfo": "K采样器高级 (含采样信息)",
}