"""
MISLG Tools - ComfyUI 自定义工具节点包
提供空输入输出节点、VAE优化、图像转换、图片切换、模型管理等实用工具
作者: MISLG
版本: 1.3.0
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
    from .model_unloader import *  # 原有的模型卸载模块
    from .model_unloader_io import *  # 新增带IO接口的模型卸载模块
    
    # 合并所有节点的映射
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
    # 从各个模块导入映射
    modules = [
        empty_input_nodes, empty_output_nodes, vae_optimizer, 
        image_converter, utils, image_switch, model_unloader, model_unloader_io  # 添加新模块
    ]
    
    for module in modules:
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    
    print(f"✅ MISLG Tools v1.3.0 已成功加载")
    print(f"   已注册 {len(NODE_CLASS_MAPPINGS)} 个节点")
    print(f"   新增功能: 带IO接口的通用模型卸载器和智能模型管理器")
    
    # 显示已加载的节点列表（可选，用于调试）
    if len(NODE_CLASS_MAPPINGS) > 0:
        print(f"   已加载节点: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
    
except ImportError as e:
    print(f"⚠️ MISLG Tools 部分模块导入失败: {e}")
    print(f"   确保所有依赖文件存在")
    # 尝试继续加载其他模块
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
    # 重新尝试导入其他模块
    try:
        from .empty_input_nodes import *
        from .empty_output_nodes import *
        from .vae_optimizer import *
        from .image_converter import *
        from .utils import *
        from .image_switch import *
        from .model_unloader import *
        
        modules = [
            empty_input_nodes, empty_output_nodes, vae_optimizer, 
            image_converter, utils, image_switch, model_unloader
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

__version__ = "1.3.0"
__author__ = "MISLG"
__description__ = "MISLG Tools - ComfyUI 自定义工具节点包"