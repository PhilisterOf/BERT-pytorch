#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WordPiece Tokenizer 训练脚本 (All-in-One 版本)

功能概述：
从预处理后的 CSV 文件中提取日志文本，训练一个领域专用的 WordPiece Tokenizer，
并生成适配 BERT-pytorch 的 vocab.txt 文件。本脚本内嵌了 LogSanitizer 类以便于独立运行。

主要步骤：
1. 读取 CSV 数据集。
2. 提取 'EventSequence' 列并保存为临时纯文本语料。
3. 定义特殊 Token (包含 BERT 标准 Token 和 Parser-Free 正则 Token)。
4. 使用 HuggingFace tokenizers 库进行训练。
5. 保存模型与词表。
"""

import os
import re # <--- [Fixed] 补充导入 re 模块
import pandas as pd
from tokenizers.implementations import BertWordPieceTokenizer

# ==================== 0. 内置工具类 (LogSanitizer) ====================

class LogSanitizer:
    """
    日志清洗器：管理正则规则，提供 parser-free 的清洗逻辑。
    此处内嵌定义是为了获取正则对应的特殊 Token 列表。
    """

    def __init__(self):
        # 正则替换规则表 (顺序很重要！)
        self.patterns = [
            # 1. 提取并替换 Block ID (最重要的会话标识)
            (r'blk_-?\d+', '[BLK]'),

            # 2. 替换 IP 地址
            (r'(\d{1,3}\.){3}\d{1,3}', '[IP]'),

            # 3. 替换 HDFS 路径
            (r'\/([\w\.\-_]+\/)+[\w\.\-_]+', '[PATH]'),

            # 4. 替换 MAC 地址或 Hex 字符串
            (r'0x[a-f0-9]+', '[HEX]'),

            # 5. 替换剩余的纯数字
            (r'\b\d+\b', '[NUM]'),

            # 6. 清洗特殊符号
            (r'\$', ' ')
        ]

        # 预编译正则以提升速度
        self.compiled_patterns = [(re.compile(p), r) for p, r in self.patterns]

    def sanitize(self, content: str) -> str:
        """应用正则清洗日志内容"""
        content = content.strip()
        for regex, replacement in self.compiled_patterns:
            content = regex.sub(replacement, content)
        return content

# ==================== 1. 语料准备函数 ====================

def prepare_corpus(csv_path: str, temp_txt_path: str, target_column: str = "EventSequence"):
    """
    将 CSV 中的特定文本列提取并保存为纯文本文件，供 Tokenizer 训练使用。

    Args:
        csv_path (str): 输入的 CSV 文件路径 (例如 train.csv)。
        temp_txt_path (str): 输出的临时 TXT 文件路径。
        target_column (str): 包含清洗后日志文本的列名。

    Returns:
        int: 提取的行数。
    """
    print(f"[-] Reading CSV from {csv_path}...")

    # 逐块读取以节省内存 (Engineering Excellence)
    # 虽然 HDFS 训练集不大，但为了兼容 Thunderbird (2亿条)，建议使用 chunksize
    chunk_size = 100000
    total_lines = 0

    # TODO: 如果你的 CSV 分隔符不是逗号（例如是 \t），请修改 sep 参数
    with open(temp_txt_path, 'w', encoding='utf-8') as f_out:
        # 使用 pandas 读取 CSV
        # usecols=[target_column] 仅读取需要的列，大幅减少内存占用
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=[target_column]):
            # 过滤掉空值 (dropna) 并转换为字符串类型
            texts = chunk[target_column].dropna().astype(str).tolist()

            for text in texts:
                # 写入纯文本，去除可能的换行符，确保一行一条日志
                f_out.write(text.strip() + "\n")

            total_lines += len(texts)
            # 动态打印进度，end='\r' 实现原地刷新
            print(f"    Processed {total_lines} lines...", end='\r')

    print(f"\n[-] Temporary corpus saved to {temp_txt_path} ({total_lines} lines)")
    return total_lines


# ==================== 2. Tokenizer 训练函数 ====================

def train_tokenizer(corpus_files: list, output_dir: str, vocab_size: int = 3000):
    """
    训练 WordPiece Tokenizer 并保存。

    Args:
        corpus_files (list): 包含纯文本语料路径的列表。
        output_dir (str): 模型保存目录。
        vocab_size (int): 目标词表大小。HDFS 建议 3000，BGL 建议 5000-10000。

    Returns:
        None
    """
    # 1. 初始化 Tokenizer
    # clean_text=True: 会自动移除控制字符
    # handle_chinese_chars=False: 日志通常是全英文，关闭此项可提升速度
    # strip_accents=True: 移除变音符号 (如 é -> e)
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=True,
        lowercase=True
    )

    # 2. 定义特殊 Token
    # BERT 必须的 5 个基础 Token
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # [Changed] 无论如何都尝试加载 LogSanitizer 的规则，因为类已经在上面定义了
    sanitizer = LogSanitizer()
    # 获取 sanitizer 中定义的所有替换目标字符串
    # patterns 格式: [(regex, replacement), ...]
    # set() 去重，list() 转列表
    regex_tokens = list(set([r for _, r in sanitizer.patterns]))
    special_tokens.extend(regex_tokens)

    print(f"[-] Special tokens to be preserved: {special_tokens}")

    # 3. 执行训练 (Training)
    print(f"[-] Training tokenizer with vocab_size={vocab_size}...")
    tokenizer.train(
        files=corpus_files,
        vocab_size=vocab_size,
        min_frequency=2,  # 最小词频：出现少于2次的子词将被丢弃
        show_progress=True,
        special_tokens=special_tokens,
        limit_alphabet=1000,
        wordpieces_prefix="##"  # BERT 标准子词前缀
    )

    # 4. 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # save_model 会生成 vocab.txt
    tokenizer.save_model(output_dir)

    print(f"[-] Tokenizer saved to: {output_dir}")
    print(f"    File 'vocab.txt' is ready for BERT-pytorch.")


# ==================== 3. 主程序入口 ====================

if __name__ == "__main__":
    # 配置路径
    # TODO: 确认你的输入文件路径是否正确，只允许用训练集建立vocab
    # 注意：这里改为了你上一轮提到的目录结构
    # INPUT_CSV = "../output/hdfs/hdfs_sequence_sanitized.csv"
    INPUT_CSV = "../output/hdfs/train.csv"

    # 临时文件路径 (中间产物)
    TEMP_CORPUS = "../output/hdfs/raw_corpus.txt"

    # 输出目录 (保存 vocab.txt 的地方)
    OUTPUT_DIR = "../output/hdfs/"

    # 词表大小配置
    # TODO: 对于 BGL 或 Thunderbird，建议将此值调大到 10000
    VOCAB_SIZE = 3000

    # 步骤 1: 准备语料
    if os.path.exists(INPUT_CSV):
        prepare_corpus(INPUT_CSV, TEMP_CORPUS, target_column="EventSequence")
    else:
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        print("Please check if '../output/hdfs/hdfs_sequence_sanitized.csv' exists.")
        exit(1)

    # 步骤 2: 训练 Tokenizer
    train_tokenizer([TEMP_CORPUS], OUTPUT_DIR, vocab_size=VOCAB_SIZE)

    # 步骤 3: 清理临时文件 (可选)
    if os.path.exists(TEMP_CORPUS):
        print(f"[-] Cleaning up temporary corpus file: {TEMP_CORPUS}")
        os.remove(TEMP_CORPUS)