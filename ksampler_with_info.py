# 文件名: ksampler_with_info.py
# 保存到: ComfyUI/custom_nodes/ksampler_with_info.py

import comfy.samplers
import comfy.sample
import latent_preview
import torch

class KSamplerWithInfo:
    """增强版K采样器 - 带详细信息输出"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "normal"}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    OUTPUT_NODE = True

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        # 使用官方API获取采样器
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        # 准备采样参数
        latent = latent_image
        latent_image = latent["samples"]
        
        # 准备噪声
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        # 准备回调函数
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # 执行采样（使用官方采样函数）
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler, scheduler, positive, negative, latent_image,
            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=noise_mask, callback=callback, 
            disable_pbar=disable_pbar, seed=seed
        )
        
        # 构建详细信息字符串
        info = f"采样器: {sampler_name}\n调度器: {scheduler}\n步数: {steps}\nCFG: {cfg}\n种子: {seed}\n降噪强度: {denoise}"
        
        # 返回结果
        out = latent.copy()
        out["samples"] = samples
        return (out, info)

class KSamplerAdvancedWithInfo:
    """增强版高级K采样器 - 带详细信息输出"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"], {"default": "enable"}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "normal"}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"], {"default": "disable"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    OUTPUT_NODE = True

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise):
        # 使用官方API获取采样器
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        # 准备采样参数
        latent = latent_image
        latent_image = latent["samples"]
        disable_noise = add_noise == "disable"
        
        # 准备噪声
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, noise_seed, batch_inds)
        
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        # 准备回调函数
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # 执行采样（使用官方高级采样函数）
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler, scheduler, positive, negative, latent_image,
            denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, 
            last_step=end_at_step, force_full_denoise=return_with_leftover_noise == "disable",
            noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed
        )
        
        # 构建详细信息字符串
        info = f"采样器: {sampler_name}\n调度器: {scheduler}\n步数: {steps}\nCFG: {cfg}\n噪声种子: {noise_seed}\n开始步: {start_at_step}\n结束步: {end_at_step}\n添加噪声: {add_noise}\n保留噪声: {return_with_leftover_noise}"
        
        # 返回结果
        out = latent.copy()
        out["samples"] = samples
        return (out, info)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "KSamplerWithInfo": KSamplerWithInfo,
    "KSamplerAdvancedWithInfo": KSamplerAdvancedWithInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerWithInfo": "KSampler (With Info)",
    "KSamplerAdvancedWithInfo": "KSampler Advanced (With Info)",
}