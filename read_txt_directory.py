import os
import glob
import random
import logging
import re
import time

logger = logging.getLogger("ComfyUI.MISLG.ReadTxtDirectory")

def _natural_sort_key(filepath):
    """自然排序键：严格按文件名中的数字大小比较，如 12 > 2, 123 > 12"""
    basename = os.path.basename(filepath)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', basename)]

class ReadTxtDirectoryNodeCN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件路径": ("STRING", {"default": "input", "multiline": False}),
                "读取顺序": (["按文件名顺序", "随机顺序"], {"default": "按文件名顺序"}),
                "起始序号": ("INT", {"default": 1, "min": 1, "tooltip": "从第几个文件开始（1为第一个）"}),
                "结束序号": ("INT", {"default": 0, "min": 0, "tooltip": "读到第几个文件结束（0表示读到末尾）"}),
                "编码格式": ("STRING", {"default": "utf-8"}),
            },
            "optional": {
                "递归扫描": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("合并文本", "文本列表", "运行状态")
    FUNCTION = "execute"
    CATEGORY = "MISLG Tools/Switches"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, True, False)

    # 🔑 核心修复：强制 ComfyUI 每次运行都重新执行 execute()，绕过输入未变化导致的缓存复用
    @classmethod
    def IS_CHANGED(cls, 文件路径, 读取顺序, 起始序号, 结束序号, 编码格式, 递归扫描):
        return time.time()

    def execute(self, 文件路径, 读取顺序, 起始序号, 结束序号, 编码格式="utf-8", 递归扫描=False):
        try:
            # 1. 路径解析
            dir_path = os.path.expanduser(文件路径)
            if not os.path.isabs(dir_path):
                comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                dir_path = os.path.abspath(os.path.join(comfy_root, dir_path))
            
            if not os.path.isdir(dir_path):
                return ("", [], f"❌ 目录不存在: {dir_path}")

            # 2. 扫描文件
            pattern = "**/*.txt" if 递归扫描 else "*.txt"
            files = glob.glob(os.path.join(dir_path, pattern), recursive=递归扫描)
            
            if not files:
                return ("", [], "⚠️ 指定目录下未找到 .txt 文件")

            # 3. 自然排序
            files.sort(key=_natural_sort_key)

            # 4. 随机控制：每次执行重新打乱（因 IS_CHANGED 已强制跳过缓存，此处必然生效）
            if 读取顺序 == "随机顺序":
                random.shuffle(files)

            # 5. 区间切片（1-based）
            total = len(files)
            start = max(1, 起始序号)
            end = 结束序号 if 结束序号 > 0 else total
            end = min(end, total)

            if start > end:
                return ("", [], f"⚠️ 起始序号({start}) 大于 结束序号({end})，未读取任何文件")

            selected_files = files[start-1:end]
            contents = []

            # 6. 读取与清洗
            for fpath in selected_files:
                try:
                    with open(fpath, "r", encoding=编码格式) as f:
                        raw_data = f.read()
                    
                    cleaned_lines = [line.strip() for line in raw_data.splitlines() if line.strip()]
                    cleaned_data = ' '.join(cleaned_lines)
                    cleaned_data = re.sub(r'\s+', ' ', cleaned_data)
                    
                    if cleaned_data:
                        contents.append(cleaned_data)
                except Exception as e:
                    logger.warning(f"读取失败 {fpath}: {e}")

            # 7. 双输出组装
            combined_content = "\n\n".join(contents)
            status_msg = f"✅ 成功加载 {len(contents)} 个文件，已启用批量迭代"
            return (combined_content, contents, status_msg)

        except Exception as e:
            logger.error(f"节点执行异常: {str(e)}")
            return ("", [], f"💥 节点执行异常: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "ReadTxtDirectoryNodeCN": ReadTxtDirectoryNodeCN
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReadTxtDirectoryNodeCN": "📁 读取TXT目录(双输出)"
}