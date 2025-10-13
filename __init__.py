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
# 导入所有模块
# ======================================================
try:
    from .empty_input_nodes import *
    from .empty_output_nodes import *
    from .vae_optimizer import *
    from .image_converter import *
    from .utils import *  # 含 MISLG 工具节点模块
    from .image_switch import *
    from .model_unloader import *      # 原有的模型卸载模块
    from .model_unloader_io import *   # 新增带IO接口的模型卸载模块
    from .instant_preview_loader import * # 即时预览图片加载器与路径助手模块
    from .ksampler_with_info import *  # ✅ 新增：采样器信息模块

    # ======================================================
    # 合并节点映射
    # ======================================================
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    # 模块列表（用于批量注册）
    modules = [
        empty_input_nodes,
        empty_output_nodes,
        vae_optimizer,
        image_converter,
        utils,              # ✅ 包含 MISLG 工具节点模块
        image_switch,
        model_unloader,
        model_unloader_io,
        instant_preview_loader, # ✅ 即时预览图片加载器与路径助手模块
        ksampler_with_info     # ✅ 新增：采样器信息模块
    ]

    for module in modules:
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

    # ======================================================
    # 输出加载信息
    # ======================================================
    print(f"✅ MISLG Tools v1.3.4 已成功加载")
    print(f"   已注册 {len(NODE_CLASS_MAPPINGS)} 个节点")
    print(f"   新增功能: MISLG 工具节点模块 (双输入自动判断、图像优先、批量合并)")
    print(f"   新增功能: 即时预览图片加载器与路径助手 (即时预览、路径管理、多模式操作)")
    print(f"   新增功能: K采样器信息模块 (带详细信息的采样器)")

    if len(NODE_CLASS_MAPPINGS) > 0:
        print(f"   已加载节点: {', '.join(NODE_CLASS_MAPPINGS.keys())}")

# ======================================================
# 异常处理
# ======================================================
except ImportError as e:
    print(f"⚠️ MISLG Tools 部分模块导入失败: {e}")
    print("   请检查依赖文件是否完整。")

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    try:
        # 尝试加载基础模块
        from .empty_input_nodes import *
        from .empty_output_nodes import *
        from .vae_optimizer import *
        from .image_converter import *
        from .utils import *
        from .image_switch import *
        from .model_unloader import *
        from .instant_preview_loader import * # 尝试加载即时预览图片加载器模块
        from .ksampler_with_info import *    # ✅ 尝试加载采样器信息模块

        modules = [
            empty_input_nodes,
            empty_output_nodes,
            vae_optimizer,
            image_converter,
            utils,
            image_switch,
            model_unloader,
            instant_preview_loader, # 尝试加载即时预览图片加载器模块
            ksampler_with_info     # ✅ 尝试加载采样器信息模块
        ]

        for module in modules:
            if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

        print(f"✅ MISLG Tools 基础功能已加载 ({len(NODE_CLASS_MAPPINGS)} 个节点)")
    except Exception as e2:
        print(f"❌ MISLG Tools 完全加载失败: {e2}")

except Exception as e:
    print(f"❌ MISLG Tools 加载失败: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# ======================================================
# 模块元信息
# ======================================================
__version__ = "1.3.4"
__author__ = "MISLG"
__description__ = "MISLG Tools - ComfyUI 自定义工具节点包 (含 MISLG 工具节点模块、即时预览图片加载器、K采样器信息模块)"