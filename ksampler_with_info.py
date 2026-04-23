"""
KSampler with Info Output
Fully compatible with official ComfyUI KSampler logic
Supports: SD1.5/SDXL/SD3/Flux/z-image/Lumina and all community models
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
                "model": ("MODEL", {"tooltip": "用于去噪的扩散模型 / The diffusion model used for denoising"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "随机种子，控制噪声生成 / Random seed for noise generation"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "采样步数，步数越多细节越丰富但速度越慢 / Number of sampling steps"}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "提示词引导系数(CFG)，越高越贴合提示词但可能过曝 / Classifier-Free Guidance scale"}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler", "tooltip": "采样器算法，影响生成风格和速度 / Sampling algorithm"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "normal", "tooltip": "调度器，控制噪声衰减曲线 / Noise schedule scheduler"}),
                "positive": ("CONDITIONING", {"tooltip": "正向提示词条件 / Positive conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "反向提示词条件 / Negative conditioning"}),
                "latent_image": ("LATENT", {"tooltip": "潜空间图像，输入待去噪的latent / Latent image to denoise"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "去噪强度，1.0为文生图，低于1.0为图生图 / Denoising strength"}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    OUTPUT_NODE = True

    DESCRIPTION = "使用指定模型和条件对潜空间图像进行去噪采样，并输出采样参数信息 / Denoises latent using model with info output"
    SEARCH_ALIASES = ["ksampler", "sampler", "采样器", "采样", "生成"]

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        # 关键修复：处理空输入情况，与官方行为一致
        if latent_image is None:
            return (None, "")
        
        latent = latent_image.copy()
        latent_samples = latent["samples"]

        # 修复空 latent 通道，确保与模型兼容
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples, latent.get("downscale_ratio_spacial", None))

        sampler = comfy.samplers.sampler_object(sampler_name)
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler, scheduler,
            positive, negative, latent_samples,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed
        )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples

        # 输出格式：带字段名称的键值对
        info = f"seed：{seed}  steps：{steps}  cfg：{cfg}  sampler：{sampler_name}  scheduler：{scheduler}  denoise：{denoise}"
        return (out, info)


class KSamplerAdvancedWithInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "用于去噪的扩散模型 / The diffusion model used for denoising"}),
                "add_noise": (["enable", "disable"], {"default": "enable", "tooltip": "是否添加初始噪声，disable用于继续采样 / Enable or disable initial noise addition"}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "噪声随机种子 / Noise random seed"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "总采样步数 / Total number of sampling steps"}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "提示词引导系数(CFG) / Classifier-Free Guidance scale"}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler", "tooltip": "采样器算法 / Sampling algorithm"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "normal", "tooltip": "调度器 / Noise schedule scheduler"}),
                "positive": ("CONDITIONING", {"tooltip": "正向提示词条件 / Positive conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "反向提示词条件 / Negative conditioning"}),
                "latent_image": ("LATENT", {"tooltip": "潜空间图像输入 / Latent image input"}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "从第几步开始采样(用于接续生成) / Step to start sampling from"}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "tooltip": "在第几步结束采样 / Step to end sampling at"}),
                "return_with_leftover_noise": (["disable", "enable"], {"default": "disable", "tooltip": "是否保留残余噪声，enable可与另一个采样器连接继续采样 / Return with leftover noise for chaining"}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    OUTPUT_NODE = True

    DESCRIPTION = "高级采样器，支持自定义起止步数和噪声保留，输出采样参数信息 / Advanced sampler with custom step range and info output"
    SEARCH_ALIASES = ["ksampler advanced", "advanced sampler", "高级采样器", "高级采样"]

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise):
        
        # 关键修复：处理空输入情况，与官方行为一致
        if latent_image is None:
            return (None, "")
        
        latent = latent_image.copy()
        latent_samples = latent["samples"]

        # 修复空 latent 通道
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples, latent.get("downscale_ratio_spacial", None))

        sampler = comfy.samplers.sampler_object(sampler_name)
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, noise_seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        force_full_denoise = (return_with_leftover_noise == "disable")
        disable_noise = (add_noise == "disable")

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler, scheduler,
            positive, negative, latent_samples,
            denoise=1.0,
            disable_noise=disable_noise,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed
        )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples

        # 输出格式：带字段名称的键值对
        actual_steps = max(0, min(steps, end_at_step) - start_at_step) if start_at_step < end_at_step else 0
        info = f"seed：{noise_seed}  steps：{actual_steps}  cfg：{cfg}  sampler：{sampler_name}  scheduler：{scheduler}  denoise：1.0"
        return (out, info)


NODE_CLASS_MAPPINGS = {
    "KSamplerWithInfo": KSamplerWithInfo,
    "KSamplerAdvancedWithInfo": KSamplerAdvancedWithInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerWithInfo": "KSampler (含采样信息)",
    "KSamplerAdvancedWithInfo": "KSampler 高级 (含采样信息)",
}