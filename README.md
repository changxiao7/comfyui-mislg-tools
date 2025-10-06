# MISLG Tools - ComfyUI 自定义工具节点

版本: 1.3.0

## 🆕 新增功能

### 通用模型卸载器 (带IO接口)
- **节点名称**: `UniversalModelUnloaderWithIO`
- **位置**: MISLG Tools/Model
- **功能**: 在工作流任意位置插入，不中断数据流的情况下卸载模型

#### 主要特性：
- ✅ 完整输入输出接口支持
- ✅ 多种模型类型选择卸载
- ✅ 三种卸载模式（激进/平衡/保守）
- ✅ 实时内存监控和报告
- ✅ 调试输出功能

#### 支持的模型类型：
- VAE 模型
- CLIP 模型  
- UNet 模型
- ControlNet 模型
- 超分模型
- 人脸模型
- 分割模型
- 深度模型
- 其他自定义模型

### 智能模型管理器
- **节点名称**: `SmartModelManager`
- **功能**: 自动内存管理和优化建议

## 📋 安装方法

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/changxiao7/comfyui-mislg-tools.git