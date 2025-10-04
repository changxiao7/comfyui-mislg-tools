"""
MISLG Tools - ComfyUI 自定义工具节点包
提供空输入输出节点、VAE优化、图像转换、图片切换等实用工具
作者: MISLG
版本: 1.1.0
"""

import os
import sys

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入所有模块
try:
    from .empty_input_nodes import *
    from .empty_output_nodes import *
    from .vae_optimizer import *
    from .image_converter import *
    from .utils import *
    from .image_switch import *
    
    # 合并所有节点的映射
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
    # 从各个模块导入映射
    modules = [
        empty_input_nodes, empty_output_nodes, vae_optimizer, 
        image_converter, utils, image_switch
    ]
    
    for module in modules:
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    
    print(f"✅ MISLG Tools v1.1.0 已成功加载")
    print(f"   已注册 {len(NODE_CLASS_MAPPINGS)} 个节点")
    
except Exception as e:
    print(f"❌ MISLG Tools 加载失败: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.1.0"
__author__ = "MISLG"
__description__ = "MISLG Tools - ComfyUI 自定义工具节点包"