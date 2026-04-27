import json
import os
import ast
import re

class ASRJsonToSubtitleConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "timestamps": ("STRING", {"default": "", "multiline": False, "tooltip": "Qwen3-ASR 输出的时间戳 (JSON/列表/[时间-时间]文本)"}),
                "text_input": ("STRING", {"multiline": True, "default": "", "tooltip": "精校文本。节点将按句子在ASR时间轴上严格锚定并平滑节奏"}),
                "output_format": (["LRC", "SRT"], {"default": "LRC", "tooltip": "输出字幕格式"}),
                "enable_optimization": ("BOOLEAN", {"default": True, "tooltip": "启用超长行标点拆分"}),
                "max_chars_per_line": ("INT", {"default": 20, "min": 5, "max": 50, "step": 1, "tooltip": "单行最大字符数"}),
                "enable_pacing_norm": ("BOOLEAN", {"default": True, "tooltip": "启用节奏归一化（自动切除ASR长静音，防止字幕快慢不均）"}),
                "time_offset": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "全局时间轴偏移(秒)"}),
            },
            "optional": {
                "output_filepath": ("STRING", {"default": "", "multiline": False, "tooltip": "留空仅输出文本流，填路径则自动保存文件"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("subtitle_content", "saved_file_path")
    FUNCTION = "process"
    CATEGORY = "MISLG Tools/Utils"
    DESCRIPTION = "句级严格锚点+节奏平滑转换器：解决ASR静音拉伸导致的快慢不均问题"

    def process(self, timestamps, text_input, output_format, enable_optimization, 
                max_chars_per_line, enable_pacing_norm, time_offset, output_filepath=""):
        
        asr_data = self._parse_timestamps(timestamps)
        if not asr_data:
            raise ValueError("未提取到有效时间戳。请检查 Qwen3-ASR 输出格式。")
        
        asr_flat = self._flatten_asr(asr_data)
        if not asr_flat:
            raise ValueError("ASR数据中未包含有效文本。")

        text = text_input.strip()
        if text:
            segments = self._align_with_pacing_smoothing(text, asr_flat, enable_pacing_norm)
        else:
            segments = self._extract_raw_segments(asr_data)

        if not segments:
            raise ValueError("未生成有效字幕片段。请提供 text_input 或检查时间戳结构。")

        # 后处理：仅拆分超长行
        if enable_optimization:
            segments = self._split_only_long_segments(segments, max_chars_per_line)

        # 全局偏移 & 防负数
        for seg in segments:
            seg['start'] = max(0.0, seg['start'] + time_offset)
            seg['end'] = max(0.0, seg['end'] + time_offset)

        # 格式化输出
        if output_format == "LRC":
            content = "\n".join(f"{self._sec_to_lrc(seg['start'])}{seg['text']}" for seg in segments)
        else:
            lines = []
            for i, seg in enumerate(segments, 1):
                lines.append(f"{i}\n{self._sec_to_srt(seg['start'])} --> {self._sec_to_srt(seg['end'])}\n{seg['text']}")
            content = "\n\n".join(lines) + "\n"

        # 保存文件
        saved_path = ""
        if output_filepath.strip():
            abs_path = os.path.abspath(output_filepath)
            dir_path = os.path.dirname(abs_path)
            if dir_path: os.makedirs(dir_path, exist_ok=True)
            with open(abs_path, 'w', encoding='utf-8') as f: f.write(content)
            saved_path = abs_path

        return (content, saved_path)

    def _align_with_pacing_smoothing(self, text, asr_flat, enable_pacing):
        """核心：前向匹配 + 静音修剪 + 节奏归一化"""
        # 1. 清理文本与索引映射
        clean_text = ''.join([c for c in text if not c.isspace()])
        asr_clean = [(self._norm(s['char']), s['start'], s['end']) for s in asr_flat if s['char'].strip()]
        
        if not clean_text or not asr_clean: return []

        # 2. 前向贪心匹配（严格单调递增，跳过ASR噪声/重复）
        matches = []  # (clean_text_idx, start, end)
        asr_ptr = 0
        for i, tc in enumerate(clean_text):
            norm_tc = self._norm(tc)
            found = False
            search_limit = min(asr_ptr + 300, len(asr_clean))
            for j in range(asr_ptr, search_limit):
                if asr_clean[j][0] == norm_tc:
                    matches.append((i, asr_clean[j][1], asr_clean[j][2]))
                    asr_ptr = j + 1
                    found = True
                    break
            if not found:
                prev = matches[-1] if matches else (0, asr_clean[0][1], asr_clean[0][2])
                matches.append((i, prev[1] + 0.1, prev[2] + 0.2))

        # 3. 按句子分组匹配结果
        sentences = self._split_text_to_sentences(text)
        char_ranges = []
        idx = 0
        for sent in sentences:
            sent_clean = ''.join([c for c in sent if not c.isspace()])
            char_ranges.append((idx, idx + len(sent_clean)))
            idx += len(sent_clean)

        segments = []
        last_end = 0.0
        for s_idx, sent in enumerate(sentences):
            if not sent.strip(): continue
            start_idx, end_idx = char_ranges[s_idx]
            sent_matches = [m for m in matches if start_idx <= m[0] < end_idx]

            if not sent_matches:
                start = last_end + 0.2
                end = start + max(0.5, len(sent.strip()) * 0.15)
            else:
                start = sent_matches[0][1]
                end = sent_matches[-1][2]
                
                # 节奏归一化：切除ASR长静音间隙
                if enable_pacing:
                    last_active = sent_matches[-1][2]
                    if end - last_active > 0.8:  # 句尾静音超过0.8秒则修剪
                        end = last_active + 0.3
                
                # 单调性保障
                start = max(start, last_end)
                end = max(end, start + 0.35)  # 最小可读时长

            segments.append({'start': start, 'end': end, 'text': sent.strip()})
            last_end = end

        # 4. 全局匹配率检查（低于35%则降级为比例分配，防异常跳变）
        match_rate = len([m for m in matches if m[1] > 0]) / max(1, len(clean_text))
        if match_rate < 0.35:
            return self._proportional_fallback(sentences, asr_flat[0]['start'], asr_flat[-1]['end'])
            
        return segments

    def _proportional_fallback(self, sentences, start_t, end_t):
        total_dur = end_t - start_t
        if total_dur <= 0: total_dur = 10.0
        current = start_t
        segments = []
        for sent in sentences:
            dur = max(0.5, len(sent.strip()) * 0.18)
            segments.append({'start': current, 'end': current + dur, 'text': sent.strip()})
            current += dur
        return segments

    def _split_text_to_sentences(self, text):
        strong_puncts = set('。！？；\n.!?;')
        sentences, current = [], []
        for char in text:
            current.append(char)
            if char in strong_puncts:
                seg = ''.join(current).strip()
                if seg: sentences.append(seg)
                current = []
        if current:
            seg = ''.join(current).strip()
            if seg: sentences.append(seg)
        return sentences

    def _flatten_asr(self, asr_data):
        flat = []
        for seg in asr_data:
            t = str(seg.get('text', '')).strip()
            for c in t:
                flat.append({'char': c, 'start': seg['start'], 'end': seg['end']})
        return flat

    def _parse_timestamps(self, ts_input):
        if isinstance(ts_input, (dict, list)): data = ts_input
        elif isinstance(ts_input, str):
            ts_input = ts_input.strip()
            if not ts_input: return []
            try: data = json.loads(ts_input)
            except json.JSONDecodeError:
                try: data = ast.literal_eval(ts_input)
                except Exception:
                    pattern = re.compile(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*(.*)')
                    segs = []
                    for line in ts_input.splitlines():
                        m = pattern.match(line.strip())
                        if m: segs.append({"start": float(m.group(1)), "end": float(m.group(2)), "text": m.group(3).strip()})
                    return segs
        else: return []

        items = data if isinstance(data, list) else []
        if isinstance(data, dict):
            for key in ["segments", "results", "data", "timestamps", "output"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            else: items = [data]

        result = []
        for item in items:
            if not isinstance(item, dict): continue
            start = (item.get("start") or item.get("begin") or item.get("start_time") or
                     (item.get("timestamp")[0] if isinstance(item.get("timestamp"), (list, tuple)) and len(item.get("timestamp")) >= 2 else None))
            end = (item.get("end") or item.get("finish") or item.get("end_time") or
                   (item.get("timestamp")[1] if isinstance(item.get("timestamp"), (list, tuple)) and len(item.get("timestamp")) >= 2 else None))
            text = item.get("text") or item.get("content") or item.get("sentence") or item.get("word") or item.get("transcript") or ""
            if start is not None and end is not None:
                result.append({"start": float(start), "end": float(end), "text": str(text)})
        return result

    def _extract_raw_segments(self, asr_data):
        return [s for s in asr_data if s.get('text','').strip()]

    def _split_only_long_segments(self, segments, max_chars):
        optimized = []
        for seg in segments:
            optimized.extend(self._split_long_segment(seg, max_chars))
        return optimized

    def _split_long_segment(self, seg, max_chars):
        if len(seg['text']) <= max_chars: return [seg]
        text = seg['text']
        puncts = [i for i, c in enumerate(text) if c in '，。！？；、,.!?;)]》”\'"']
        valid_puncts = [p for p in puncts if p < max_chars]
        split_idx = valid_puncts[-1] if valid_puncts else min(max_chars, len(text)-1)
        if split_idx <= 0: return [seg]
        part1, part2 = text[:split_idx+1].strip(), text[split_idx+1:].strip()
        if not part1 or not part2: return [seg]
        duration = seg['end'] - seg['start']
        ratio = len(part1) / len(text)
        split_time = seg['start'] + duration * ratio
        return [
            {"start": seg['start'], "end": split_time, "text": part1},
            {"start": split_time, "end": seg['end'], "text": part2}
        ]

    @staticmethod
    def _norm(c):
        mapping = {'，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':', '“': '"', '”': '"', '‘': "'", '’': "'"}
        return mapping.get(c, c)

    @staticmethod
    def _sec_to_lrc(sec): return f"[{int(sec//60):02d}:{sec%60:05.2f}]"
    @staticmethod
    def _sec_to_srt(sec): return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{sec%60:06.3f}".replace('.', ',')

NODE_CLASS_MAPPINGS = {"ASRJsonToSubtitleConverter": ASRJsonToSubtitleConverter}
NODE_DISPLAY_NAME_MAPPINGS = {"ASRJsonToSubtitleConverter": "🎤 句级锚点+节奏平滑转换器"}