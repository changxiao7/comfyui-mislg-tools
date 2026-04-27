"""
MISLG Tools - ComfyUI 自定义工具节点包
提供空输入输出节点、VAE优化、图像转换、图片切换、模型管理等实用工具
作者: MISLG
版本: 1.3.4
"""
import os
import sys

# ======================================================
# 初始化路径
# ======================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ======================================================
# 初始化映射字典
# ======================================================
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ======================================================
# 导入所有模块并注册
# ======================================================
try:
    from .empty_input_nodes import *
    from .empty_output_nodes import *
    from .vae_optimizer import *
    from .image_converter import *
    from .utils import *
    from .image_switch import *
    from .model_unloader import *
    from .model_unloader_io import *
    from .instant_preview_loader import *
    from .ksampler_with_info import *
    from .asr_subtitle_converter import *  # ✅ 新增：ASR 字幕转换器

    modules = [
        empty_input_nodes,
        empty_output_nodes,
        vae_optimizer,
        image_converter,
        utils,
        image_switch,
        model_unloader,
        model_unloader_io,
        instant_preview_loader,
        ksampler_with_info,
        asr_subtitle_converter  # ✅ 注册新模块
    ]

    for module in modules:
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

    print(f"✅ MISLG Tools v1.3.4 已成功加载")
    print(f"   已注册 {len(NODE_CLASS_MAPPINGS)} 个节点")
    print(f"   包含功能: 空节点/VAE优化/图像转换/图片切换/模型卸载/即时预览/K采样器信息/ASR字幕转换")
    if len(NODE_CLASS_MAPPINGS) > 0:
        print(f"   节点列表: {', '.join(NODE_CLASS_MAPPINGS.keys())}")

except ImportError as e:
    print(f"⚠️ MISLG Tools 部分模块导入失败: {e}")
    print("   请检查依赖文件是否完整。")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
except Exception as e:
    print(f"❌ MISLG Tools 加载失败: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# ======================================================
# 模块元信息
# ======================================================
version = "1.3.4"
author = "MISLG"
description = "MISLG Tools - ComfyUI 自定义工具节点包 (含 ASR 字幕转换器、即时预览、K采样器信息等)"