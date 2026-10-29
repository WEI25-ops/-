#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
英语句子分析器 (English Sentence Analyzer)
==========================================

功能：
    1. 读取文件路径，自动检测语言（中文/英文）
    2. 中英文对照翻译（上排中文，下排英文）
    3. 拆解英文句子语法成分（主谓宾定状补）
    4. 判断句子类型（陈述句/疑问句/祈使句/感叹句等）
    5. 识别句型结构并按出现次数从高到低排列

依赖：
    pip install spacy deep-translator nltk
    python -m spacy download en_core_web_sm

使用方法：
    python english_analyzer.py <文件路径>
    python english_analyzer.py ./sample.txt
"""

import os
import re
import sys
import json
import time
import requests
from collections import Counter
from typing import List, Dict, Tuple, Optional

import spacy


# =============================================================================
# 配置
# =============================================================================

# 代理配置（如无代理可设为空字符串或None）
PROXIES = {
    "http": os.environ.get("HTTP_PROXY", "http://127.0.0.1:18080"),
    "https": os.environ.get("HTTPS_PROXY", "http://127.0.0.1:18080"),
}

# 仅在代理可用时使用
USE_PROXY = bool(PROXIES.get("https") or PROXIES.get("http"))

# 初始化 spaCy 英文模型（全局加载，避免重复加载）
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("[错误] 未找到 spacy 英文模型 en_core_web_sm")
    print("请先运行: python -m spacy download en_core_web_sm")
    sys.exit(1)


# =============================================================================
# 工具函数
# =============================================================================

def detect_language(text: str) -> str:
    """
    检测文本主要语言。
    返回 'zh'（中文）或 'en'（英文）。
    """
    # 统计中文字符数
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 统计英文字母数
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    if zh_chars > en_chars:
        return "zh"
    return "en"


def split_sentences(text: str, lang: str = "en") -> List[str]:
    """
    将文本切分成句子列表，保留原始标点。
    中文和英文使用不同的切分策略。
    """
    text = text.strip()
    if not text:
        return []

    if lang == "en":
        # 英文：按句号、问号、感叹号切分，保留标点
        sentences = re.split(r'(?<=[.!?])\s+', text)
    else:
        # 中文：按句号、问号、感叹号切分
        sentences = re.split(r'(?<=[。！？])\s*', text)

    # 清理空句子并去除首尾空白
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def read_file(file_path: str) -> str:
    """读取文件内容，自动尝试多种编码。"""
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        sys.exit(1)

    # 尝试常见编码
    encodings = ["utf-8", "gbk", "gb18030", "utf-16"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            print(f"[信息] 文件读取成功（编码: {enc}）")
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue

    print("[错误] 无法识别文件编码，请将文件转换为 UTF-8 格式后重试。")
    sys.exit(1)


# =============================================================================
# 翻译模块（基于 MyMemory 免费 API）
# =============================================================================

class Translator:
    """基于 MyMemory 免费 API 的翻译器，支持代理。"""

    API_URL = "https://api.mymemory.translated.net/get"

    def __init__(self, use_proxy: bool = True):
        self.proxies = PROXIES if use_proxy else None
        self._cache: Dict[str, str] = {}  # 简单缓存，避免重复请求
        self._rate_limit_delay = 0.3  # MyMemory 限速：建议每句间隔

    def translate(self, text: str, src: str = "en", dest: str = "zh-CN") -> str:
        """翻译单句，带缓存和重试。"""
        if not text.strip():
            return text

        cache_key = f"{src}|{dest}|{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {"q": text, "langpair": f"{src}|{dest}"}

        for attempt in range(3):  # 最多重试3次
            try:
                resp = requests.get(
                    self.API_URL,
                    params=params,
                    proxies=self.proxies if USE_PROXY else None,
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    translated = data.get("responseData", {}).get("translatedText", "")
                    if translated:
                        self._cache[cache_key] = translated
                        return translated
                # 非200或无内容，重试
                time.sleep(self._rate_limit_delay)
            except Exception as e:
                if attempt == 2:
                    print(f"  [警告] 翻译失败: {text[:30]}... - {e}")
                    return "[翻译失败]"
                time.sleep(self._rate_limit_delay * (attempt + 1))

        return "[翻译失败]"

    def translate_batch(self, sentences: List[str], src: str = "en", dest: str = "zh-CN") -> List[str]:
        """批量翻译句子，带进度提示。"""
        results = []
        total = len(sentences)
        for i, sent in enumerate(sentences):
            if sent.strip():
                translated = self.translate(sent, src, dest)
            else:
                translated = sent
            results.append(translated)
            # 每10句打印一次进度
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"  [翻译进度] {i + 1}/{total}")
            # 限速
            time.sleep(self._rate_limit_delay)
        return results


# =============================================================================
# 句子类型判断
# =============================================================================

def detect_sentence_type(doc: spacy.tokens.Doc) -> str:
    """
    判断英文句子类型。

    返回值：
        - 陈述句 (Declarative)
        - 一般疑问句 (Interrogative - Yes/No)
        - 特殊疑问句 (Interrogative - Wh-)
        - 祈使句 (Imperative)
        - 感叹句 (Exclamatory)
        - 其他 (Other)
    """
    text = doc.text.strip()

    # 检查结尾标点
    last_char = text[-1] if text else ""
    if last_char == "!":
        # 感叹句
        return "感叹句 (Exclamatory)"

    if last_char == "?":
        # 疑问句：检查句首是否为疑问词
        first_token = doc[0].text.lower() if len(doc) > 0 else ""
        wh_words = {"what", "who", "whom", "whose", "which", "when", "where", "why", "how"}
        if first_token in wh_words:
            return "特殊疑问句 (Interrogative - Wh)"
        return "一般疑问句 (Interrogative - Yes/No)"

    # 祈使句判断：无主语，动词原形开头
    if len(doc) > 0:
        first_tok = doc[0]
        # 检查根词是否为动词，且无显式主语（祈使句省略 you）
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break
        if root:
            # 祈使句：根词为动词原形，且句子中没有作为主语的名词
            if (root.pos_ == "VERB" and
                root.tag_ == "VB" and  # 动词原形
                not any(t.dep_ in ("nsubj", "nsubjpass") for t in doc)):
                return "祈使句 (Imperative)"

    return "陈述句 (Declarative)"


# =============================================================================
# 句子成分分析（主谓宾定状补）
# =============================================================================

# spaCy 依赖标签 → 中文成分名映射
DEP_LABEL_MAP = {
    # 主语
    "nsubj": "主语(主格)",
    "nsubjpass": "主语(被动)",
    "csubj": "主语(从句)",
    # 谓语（根词通常是谓语动词）
    "ROOT": "谓语/核心",
    # 宾语
    "dobj": "直接宾语",
    "iobj": "间接宾语",
    "pobj": "介词宾语",
    "obj": "宾语",
    # 定语
    "det": "限定词(定语)",
    "amod": "形容词(定语)",
    "nmod": "名词(定语)",
    "poss": "所有格(定语)",
    "acl": "关系从句(定语)",
    "acomp": "表语补足语(定语)",
    "appos": "同位语(定语)",
    # 状语
    "advmod": "副词(状语)",
    "prep": "介词短语(状语)",
    "npadvmod": "名词短语(状语)",
    "aux": "助动词",
    "mark": "引导词",
    # 补语
    "xcomp": "外置补语",
    "ccomp": "从句补语",
    "complement": "补语",
    "attr": "表语",
    # 并列/连接
    "conj": "并列成分",
    "cc": "连词",
    # 否定
    "neg": "否定词",
    # 标点等
    "punct": "标点",
    "dep": "依存",
}


def _merge_spans(doc: spacy.tokens.Doc, tokens: List[spacy.tokens.Token]) -> List[str]:
    """将连续token合并成文本片段。"""
    if not tokens:
        return []

    # 按位置排序
    tokens = sorted(tokens, key=lambda t: t.i)
    spans = []
    current_start = tokens[0].i
    current_end = tokens[0].i

    for t in tokens[1:]:
        if t.i == current_end + 1:
            current_end = t.i
        else:
            spans.append(doc[current_start:current_end + 1].text)
            current_start = t.i
            current_end = t.i
    spans.append(doc[current_start:current_end + 1].text)
    return spans


def analyze_syntax(doc: spacy.tokens.Doc) -> Dict[str, List[str]]:
    """
    分析句子的语法成分，归类为主谓宾定状补。

    返回：
        {
            "主语": [...],
            "谓语": [...],
            "宾语": [...],
            "定语": [...],
            "状语": [...],
            "补语": [...],
            "表语": [...],
            "其他": [...],
        }
    """
    components = {
        "主语": [],
        "谓语": [],
        "宾语": [],
        "定语": [],
        "状语": [],
        "补语": [],
        "表语": [],
        "其他": [],
    }

    for token in doc:
        dep = token.dep_

        # 跳过标点
        if dep == "punct":
            continue

        text = token.text_with_ws.strip() or token.text

        if dep in ("nsubj", "nsubjpass", "csubj"):
            components["主语"].append(text)
        elif dep == "ROOT":
            components["谓语"].append(text)
        elif dep in ("dobj", "iobj", "obj"):
            components["宾语"].append(text)
        elif dep == "pobj":
            # 介词宾语通常也是句子宾语的一部分
            components["宾语"].append(text)
        elif dep in ("det", "amod", "nmod", "poss", "acl", "appos"):
            components["定语"].append(text)
        elif dep in ("advmod", "prep", "npadvmod"):
            components["状语"].append(text)
        elif dep in ("xcomp", "ccomp", "compound"):
            components["补语"].append(text)
        elif dep in ("attr", "acomp"):
            components["表语"].append(text)
        elif dep == "neg":
            # 否定词通常归为状语
            components["状语"].append(text)
        elif dep in ("aux", "mark", "cc", "conj"):
            # 辅助成分，合并到对应的主要成分中
            # 简单处理：连词归其他，助动词归谓语附近
            if dep == "cc":
                components["其他"].append(text)
            elif dep == "conj":
                components["其他"].append(text)
            else:
                components["其他"].append(f"[{DEP_LABEL_MAP.get(dep, dep)}] {text}")
        else:
            components["其他"].append(f"[{DEP_LABEL_MAP.get(dep, dep)}] {text}")

    return components


def format_components(components: Dict[str, List[str]]) -> str:
    """将成分字典格式化为可读字符串。"""
    lines = []
    labels = ["主语", "谓语", "宾语", "定语", "状语", "补语", "表语"]
    for label in labels:
        items = components.get(label, [])
        if items:
            lines.append(f"  {label}: {', '.join(items)}")
    if components.get("其他"):
        lines.append(f"  其他: {', '.join(components['其他'])}")
    return "\n".join(lines)


# =============================================================================
# 句型结构识别
# =============================================================================

def extract_pattern(doc: spacy.tokens.Doc) -> Tuple[str, Dict]:
    """
    从 spaCy 依赖解析结果中提取句型结构。

    返回 (模式字符串, 详情字典)
    模式示例: "SV", "SVO", "SVOA", "SVA", "SVOC", "SVC" 等
    """
    has_subject = any(t.dep_ in ("nsubj", "nsubjpass", "csubj") for t in doc)
    has_object = any(t.dep_ in ("dobj", "iobj", "obj") for t in doc)
    has_complement = any(t.dep_ in ("xcomp", "ccomp") for t in doc)
    has_attr = any(t.dep_ in ("attr", "acomp") for t in doc)  # 表语
    has_adverbial = any(t.dep_ in ("advmod", "prep", "npadvmod") for t in doc)

    # 检查是否为被动句
    is_passive = any(t.dep_ == "nsubjpass" for t in doc)

    # 检查是否有间接宾语（双宾语）
    has_iobj = any(t.dep_ == "iobj" for t in doc)

    # 检查祈使句（无主语）
    is_imperative = False
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    if root and root.pos_ == "VERB" and root.tag_ == "VB" and not has_subject:
        is_imperative = True

    # 确定句型代码
    pattern_parts = []

    if is_imperative:
        # 祈使句：无主语，以V开头
        pattern_parts.append("V")
    else:
        if has_subject or is_passive:
            pattern_parts.append("S")
        else:
            pattern_parts.append("_")
        # 只有非祈使句才在这里加V，祈使句的V已经加过了
        pattern_parts.append("V")

    if has_iobj:
        pattern_parts.append("IO")  # 间接宾语

    if has_object:
        pattern_parts.append("O")  # 直接宾语

    if has_attr:
        pattern_parts.append("C")  # 表语（主系表结构）

    if has_complement:
        pattern_parts.append("OC")  # 宾语补足语

    if has_adverbial:
        pattern_parts.append("A")  # 状语

    pattern_code = " → ".join(pattern_parts)

    # 根据被动/祈使等附加信息
    extras = []
    if is_passive:
        extras.append("被动语态")
    if is_imperative:
        extras.append("祈使句")

    pattern_detail = {
        "code": pattern_code,
        "is_passive": is_passive,
        "is_imperative": is_imperative,
        "extras": extras,
    }

    return pattern_code, pattern_detail


def pattern_to_description(pattern_code: str) -> str:
    """将句型代码转换为中文描述。"""
    # 句型描述映射
    desc_map = {
        "S → V": "主谓结构 (S+V)",
        "S → V → A": "主谓状结构 (S+V+A)",
        "S → V → O": "主谓宾结构 (S+V+O)",
        "S → V → O → A": "主谓宾状结构 (S+V+O+A)",
        "S → V → IO → O": "主谓双宾结构 (S+V+IO+O)",
        "S → V → IO → O → A": "主谓双宾状结构",
        "S → V → C": "主系表结构 (S+V+C)",
        "S → V → C → A": "主系表状结构 (S+V+C+A)",
        "S → V → O → OC": "主谓宾宾补结构 (S+V+O+OC)",
        "S → V → O → OC → A": "主谓宾宾补状结构",
        "_ → V": "祈使句 (V)",
        "_ → V → A": "祈使句状 (V+A)",
        "_ → V → O": "祈使句宾 (V+O)",
        "_ → V → O → A": "祈使句宾状 (V+O+A)",
        # 祈使句（无主语，直接以V开头）
        "V": "祈使句 (V)",
        "V → A": "祈使句+状语 (V+A)",
        "V → O": "祈使句+宾语 (V+O)",
        "V → O → A": "祈使句+宾+状 (V+O+A)",
        # 带补语/表语的复杂句型
        "S → V → OC": "主谓宾补结构 (S+V+OC)",
        "S → V → OC → A": "主谓宾补状结构",
        "S → V → O → OC": "主谓宾宾补结构 (S+V+O+OC)",
        "S → V → O → OC → A": "主谓宾宾补状结构",
        "S → V → O → C → OC": "主谓宾表补混合",
    }
    return desc_map.get(pattern_code, f"其他: {pattern_code}")


# =============================================================================
# 主处理流程
# =============================================================================

def process_document(file_path: str, output_path: Optional[str] = None) -> None:
    """
    主处理函数：读取文件 → 翻译 → 语法分析 → 统计 → 输出结果。
    """
    print("=" * 60)
    print("  英语句子分析器 English Sentence Analyzer")
    print("=" * 60)

    # 1. 读取文件
    print(f"\n[1/5] 读取文件: {file_path}")
    content = read_file(file_path)

    # 2. 检测语言 & 切分句子
    src_lang = detect_language(content)
    dest_lang = "zh-CN" if src_lang == "en" else "en"
    lang_name = "英文" if src_lang == "en" else "中文"
    print(f"[2/5] 检测源语言: {lang_name}")

    sentences = split_sentences(content, src_lang)
    print(f"      共检测到 {len(sentences)} 个句子")

    if not sentences:
        print("[提示] 未检测到有效句子，程序结束。")
        return

    # 3. 翻译
    print("[3/5] 执行翻译...")
    translator = Translator(use_proxy=USE_PROXY)
    # 只翻译非空句子
    translations = []
    for i, sent in enumerate(sentences):
        # 检测句子本身是否已经是混合语言
        sent_lang = detect_language(sent)
        if sent_lang == dest_lang or (src_lang == "zh" and sent_lang == "zh"):
            # 已存在目标语言，跳过翻译
            translations.append(translator.translate(sent, src=src_lang, dest=dest_lang))
        else:
            translations.append(translator.translate(sent, src=src_lang, dest=dest_lang))

    # 4. 语法分析（仅针对英文句子）
    print("[4/5] 语法分析...")
    print("-" * 60)

    sentence_types = []  # 句子类型列表
    pattern_counter: Counter = Counter()  # 句型统计

    analysis_results = []

    for idx, (sent, trans) in enumerate(zip(sentences, translations), 1):
        # 语法分析只针对英文句子
        # 如果源是中文，翻译后的英文才是分析对象
        analyze_text = sent if src_lang == "en" else trans

        # 分析类型
        print(f"\n【句 {idx}】")
        # 上排中文，下排英文
        if src_lang == "en":
            print(f"  中文: {trans}")
            print(f"  英文: {sent}")
        else:
            print(f"  中文: {sent}")
            print(f"  英文: {trans}")

        # 只对有效英文进行语法分析
        if analyze_text and analyze_text.strip() and "[翻译失败]" not in analyze_text:
            try:
                doc = NLP(analyze_text.strip())

                # 句子类型
                sent_type = detect_sentence_type(doc)
                sentence_types.append(sent_type)
                print(f"  类型: {sent_type}")

                # 语法成分
                components = analyze_syntax(doc)
                comp_str = format_components(components)
                if comp_str:
                    print(f"  成分:\n{comp_str}")

                # 句型结构
                pattern_code, pattern_detail = extract_pattern(doc)
                pattern_desc = pattern_to_description(pattern_code)

                # 附加信息
                extras_str = ""
                if pattern_detail["extras"]:
                    extras_str = " [(" + ", ".join(pattern_detail["extras"]) + ")]"

                print(f"  句型: {pattern_desc}{extras_str}")
                pattern_counter[pattern_code] += 1

                analysis_results.append({
                    "index": idx,
                    "source": sent,
                    "translation": trans,
                    "type": sent_type,
                    "components": components,
                    "pattern_code": pattern_code,
                    "pattern_desc": pattern_desc,
                    "pattern_extras": pattern_detail["extras"],
                })
            except Exception as e:
                print(f"  [警告] 语法分析失败: {e}")
        else:
            print(f"  [跳过] 无法进行语法分析")

    # 5. 句型统计
    print("\n" + "=" * 60)
    print("【5/5】句型统计 (按出现次数从高到低)")
    print("=" * 60)

    if pattern_counter:
        print(f"\n{'排名':<6}{'句型代码':<30}{'中文描述':<30}{'次数':<8}")
        print("-" * 74)
        for rank, (code, count) in enumerate(pattern_counter.most_common(), 1):
            desc = pattern_to_description(code)
            print(f"{rank:<6}{code:<30}{desc:<30}{count:<8}")
        print("-" * 74)
        print(f"{'合计':<6}{len(pattern_counter):<30}{'种句型':<30}{sum(pattern_counter.values()):<8}")

    # 句子类型统计
    if sentence_types:
        print(f"\n句子类型分布:")
        type_counter = Counter(sentence_types)
        for t, c in type_counter.most_common():
            print(f"  {t}: {c} 次")

    # 可选：输出到文件
    if output_path:
        print(f"\n结果已输出到: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for r in analysis_results:
                f.write(f"\n【句 {r['index']}】\n")
                if src_lang == "en":
                    f.write(f"  中文: {r['translation']}\n")
                    f.write(f"  英文: {r['source']}\n")
                else:
                    f.write(f"  中文: {r['source']}\n")
                    f.write(f"  英文: {r['translation']}\n")
                f.write(f"  类型: {r['type']}\n")
                f.write(f"  句型: {r['pattern_desc']}{' [(' + ', '.join(r['pattern_extras']) + ')]' if r['pattern_extras'] else ''}\n")
                comp_str = format_components(r['components'])
                if comp_str:
                    f.write(f"  成分:\n{comp_str}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("句型统计 (按出现次数从高到低)\n")
            f.write("=" * 60 + "\n")
            for rank, (code, count) in enumerate(pattern_counter.most_common(), 1):
                desc = pattern_to_description(code)
                f.write(f"  {rank}. {code} - {desc}: {count} 次\n")

    print("\n✓ 分析完成！")


# =============================================================================
# 入口
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("用法: python english_analyzer.py <文件路径> [--output 输出路径]")
        print("示例: python english_analyzer.py ./article.txt")
        print("      python english_analyzer.py ./article.txt --output result.txt")
        sys.exit(1)

    file_path = sys.argv[1]
    output_path = None

    # 解析可选参数
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]

    process_document(file_path, output_path)


if __name__ == "__main__":
    main()
