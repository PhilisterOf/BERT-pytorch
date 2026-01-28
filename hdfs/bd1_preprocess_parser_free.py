#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LogBERT HDFS 数据预处理脚本 (Parser-Free 版本)

本脚本实现了基于正则表达式的日志清洗流程，替代了传统的 Drain 解析器。
核心目标是将非结构化的原始日志转化为保留语义的文本序列，用于 BERT 模型的输入。

主要步骤：
1. 定义正则表达式清洗规则 (LogSanitizer)。
2. 读取原始 HDFS 日志，提取 BlockId 作为会话标识。
3. 清洗日志内容，替换 IP、BlockId 等高频动态变量。
4. 按 BlockId 聚合日志序列。
5. 关联异常标签 (Anomaly Label)。
6. 划分训练集与测试集 (Train/Test Split)。

文件名	格式	内容含义	核心用途
hdfs_sequence_sanitized.csv	CSV	全量数据母版。包含所有清洗后的日志、BlockId 和 Label。	存档/追溯。这是数据处理的源头，如果后续划分逻辑变了，从这里重新读取即可。
train.csv	CSV	仅包含正常样本 (Normal) 的训练集。包含 BlockId 等元数据。	模型训练。LogBERT 是无监督/自监督学习，只能看正常数据。
train.txt	TXT	仅包含日志文本。去除了 BlockId 和表头。	Tokenizer 训练。train_tokenizer.py 需要喂给它纯文本来学习词表。
test_normal.csv	CSV	未参与训练的正常样本。	计算误报率 (False Positive)。模型应该认为这些也是正常的。
test_abnormal.csv	CSV	所有的异常样本 (Anomaly)。	计算召回率 (Recall)。模型应该报警。
"""

import sys
import os
import re
from collections import defaultdict

# 第三方库依赖
# 需要安装: pip install pandas tqdm scikit-learn
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==================== 0. 全局配置与常量定义 ====================

# 输入数据目录 (请根据实际环境修改)
# HDFS 数据集通常包含 HDFS.log (原始日志) 和 anomaly_label.csv (标签文件)
INPUT_DIR = os.path.expanduser('../data/hdfs/')  # 请根据实际路径修改

# 输出数据目录
# 处理后的 CSV 和 TXT 文件将保存在此处
OUTPUT_DIR = '../output/hdfs/'

# 输入文件名
LOG_FILE = "HDFS.log"
LABEL_FILE = "anomaly_label.csv"

# 确保输出目录存在，如果不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 1. 日志清洗类 (LogSanitizer) ====================

class LogSanitizer:
    """
    日志清洗器类。

    该类负责管理正则表达式规则，并将原始日志文本中的动态变量（如 IP 地址、
    Block ID、Hex 字符串等）替换为统一的特殊 Token。这是 Parser-Free 方法的核心，
    旨在防止 BERT 词表（Vocabulary）因动态变量而爆炸（OOV 问题）。
    """

    def __init__(self):
        """
        初始化 LogSanitizer，编译正则表达式以提升匹配性能。

        Args:
            无

        Returns:
            无
        """
        # 定义正则替换规则表。
        # 注意：列表顺序至关重要，正则匹配是按顺序执行的。
        # 必须先处理长模式（如 IP、Path），再处理短模式（如纯数字），
        # 否则可能会破坏长模式的结构。
        self.patterns = [
            # 1. 提取并替换 Block ID (HDFS 中的核心实体)
            # 模式解释：匹配 "blk_" 开头，可选的负号 "-"，后跟一串数字
            # 示例：blk_12345, blk_-98765 -> [BLK]
            (r'blk_-?\d+', '[BLK]'),

            # 2. 替换 IP 地址 (IPv4)
            # 模式解释：匹配三组 (1-3位数字加点)，最后跟一组1-3位数字
            # 示例：10.10.1.1 -> [IP]
            (r'(\d{1,3}\.){3}\d{1,3}', '[IP]'),

            # 3. 替换 HDFS 文件路径
            # 模式解释：匹配以 "/" 开头，包含字母/数字/点/划线的路径结构
            # 示例：/mnt/hdfs/data -> [PATH]
            (r'\/([\w\.\-_]+\/)+[\w\.\-_]+', '[PATH]'),

            # 4. 替换 十六进制字符串
            # 模式解释：匹配 "0x" 开头后跟 16 进制字符
            # 示例：0x0000a1 -> [HEX]
            (r'0x[a-f0-9]+', '[HEX]'),

            # 5. 替换剩余的纯数字
            # 模式解释：匹配单词边界内的连续数字
            # 说明：这是为了处理耗时 (200ms)、大小 (1024B) 等数值信息
            # 注意：必须在 IP 和 Block ID 处理之后执行，防止误伤
            # 示例：200 -> [NUM]
            (r'\b\d+\b', '[NUM]'),

            # 6. 清洗特殊符号
            # 说明：Java 日志中常见 "类名$内部类"，如 "DataNode$PacketResponder"
            # BERT 的 Tokenizer 通常不处理 "$"，将其替换为空格有助于分词
            # 示例：DataNode$PacketResponder -> DataNode PacketResponder
            (r'\$', ' ')
        ]

        # 预编译正则对象
        # 列表推导式 (List Comprehension) 用于生成 (Pattern对象, 替换串) 的元组列表
        # 编译后的正则对象在重复调用时比 re.sub(pattern, ...) 更快
        self.compiled_patterns = [(re.compile(p), r) for p, r in self.patterns]

    def sanitize(self, content: str) -> str:
        """
        应用所有正则规则清洗日志内容。

        Args:
            content (str): 原始日志的文本内容部分 (Content)。

        Returns:
            str: 清洗后的日志文本，其中动态变量已被特殊 Token 替换。
        """
        # 去除字符串首尾的空白字符 (空格、换行符等)
        content = content.strip()

        # 遍历所有编译好的正则规则，依次应用替换
        for regex, replacement in self.compiled_patterns:
            content = regex.sub(replacement, content)

        return content


# ==================== 2. 主预处理函数 (hdfs_preprocess_no_parser) ====================

def hdfs_preprocess_no_parser(log_file_path: str, label_file_path: str, output_dir: str) -> pd.DataFrame:
    """
    读取原始 HDFS 日志，执行 Parser-Free 清洗，并按 BlockId 聚合成会话序列。

    数据处理流程形状变化说明：
    1. Raw Text File: (N_lines, ) -> 每一行是一个长字符串
    2. Parsing: String -> (BlockId, Sanitized_Content)
    3. Grouping: -> Dict[BlockId, List[Sanitized_Content]]
    4. DataFrame: -> (N_sessions, 3) columns=['BlockId', 'EventSequence', 'Label']

    Args:
        log_file_path (str): HDFS.log 文件的绝对路径。
        label_file_path (str): anomaly_label.csv 文件的绝对路径。
        output_dir (str): 处理结果的保存目录。

    Returns:
        pd.DataFrame: 包含完整数据集的 DataFrame，包含 'BlockId', 'EventSequence', 'Label' 列。
    """
    print(f"[-] Processing raw logs from: {log_file_path}")

    # 实例化清洗器
    sanitizer = LogSanitizer()

    # 使用 defaultdict(list) 存储会话
    # 说明：defaultdict 是 dict 的子类，当访问不存在的 key 时，
    # 会自动调用 list() 创建一个空列表，避免了 if key not in dict 的判断逻辑。
    # 结构：{ 'blk_123': ['log1', 'log2', ...], ... }
    session_dict = defaultdict(list)

    # 预编译用于提取 Key (BlockId) 的正则
    # HDFS 日志通常通过 Block ID 来关联上下文
    blk_id_extractor = re.compile(r'(blk_-?\d+)')

    # 统计总行数，用于 tqdm 进度条显示
    # 这是一个 I/O 密集型操作，但在大文件处理中为了用户体验是值得的
    total_lines = 0
    with open(log_file_path, 'r', errors='ignore') as f:
        for _ in f:
            total_lines += 1

    # 再次打开文件进行逐行处理
    # 使用 'errors=ignore' 防止因非 UTF-8 字符导致的解码错误
    with open(log_file_path, 'r', errors='ignore') as f:
        # tqdm 用于在终端显示进度条
        for line in tqdm(f, total=total_lines, desc="Sanitizing & Grouping"):
            line = line.strip()
            # 跳过空行
            if not line:
                continue

            # 1. 解析 HDFS 日志格式
            # 典型格式: 081109 203518 143 INFO dfs.DataNode$PacketResponder: PacketResponder ...
            # 我们只需要第一个冒号后面的内容 (Content)
            # maxsplit=1 表示只分割第一次出现的冒号
            parts = line.split(':', 1)
            if len(parts) < 2:
                continue

            content = parts[1]

            # 2. 提取 Session Key (Block ID)
            # 使用 findall 查找当前行包含的所有 Block ID
            block_ids = blk_id_extractor.findall(line)
            if not block_ids:
                # 如果一行日志不包含 Block ID，则无法归属到特定会话，跳过
                continue

            # 3. 清洗语义 (Sanitize)
            # 将原始文本转换为 [BLK], [IP], [NUM] 等 Token 的序列
            clean_content = sanitizer.sanitize(content)

            # 4. 归组 (Grouping)
            # 一条日志可能涉及多个 Block (例如复制操作)，需要添加到所有相关 Block 的会话中
            for blk_id in set(block_ids):
                session_dict[blk_id].append(clean_content)

    print(f"[-] Total sessions (blocks) extracted: {len(session_dict)}")

    # ==================== Label Matching (标签匹配) ====================
    print(f"[-] Loading labels from: {label_file_path}")

    # 读取标签文件
    # label_df shape: (N_blocks, 2) -> columns: [BlockId, Label]
    label_df = pd.read_csv(label_file_path)

    # 构建 Label 字典映射，将字符串标签转换为二分类整数
    # zip() 函数将两列数据打包成元组迭代器
    # 逻辑：Anomaly -> 1, Normal -> 0
    label_dict = dict(zip(
        label_df['BlockId'],
        label_df['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    ))

    # ==================== Dataset Construction (构建数据集) ====================
    # 将会话列表转换为 BERT 友好的格式
    dataset = []

    # 设定 BERT 输入的特殊分隔符
    # 这里的 [SEP] 是 HuggingFace BERT 默认的句子分隔符
    # 目的：明确区分时序上的每一条日志，帮助模型学习“事件”的边界
    sep_token = " [SEP] "

    for blk_id, logs in tqdm(session_dict.items(), desc="Building Dataset"):
        # 过滤掉标签文件中不存在的 Block ID
        if blk_id not in label_dict:
            continue

        # 将列表中的多条日志拼接成一个长字符串
        # 示例：log1 [SEP] log2 [SEP] log3
        sequence_text = sep_token.join(logs)

        # 构建样本字典
        dataset.append({
            "BlockId": blk_id,
            "EventSequence": sequence_text,  # 这里的输入是自然语言文本，不再是模板 ID
            "Label": label_dict[blk_id]
        })

    # 转换为 DataFrame
    # df_dataset shape: (N_valid_sessions, 3)
    df_dataset = pd.DataFrame(dataset)

    # 保存完整全量数据到 CSV
    full_output_path = os.path.join(output_dir, "hdfs_sequence_sanitized.csv")
    # index=False 表示不保存行索引号
    df_dataset.to_csv(full_output_path, index=False)
    print(f"[-] Full sanitized dataset saved to: {full_output_path}")

    return df_dataset


# ==================== 3. 数据集划分函数 (split_train_test) ====================

def split_train_test(df: pd.DataFrame, output_dir: str, train_size: int = 4855, random_state: int = 42):
    """
    将数据集划分为训练集和测试集。

    划分策略遵循 DeepLog/LogBERT 的标准设定：
    1. 训练集仅包含正常样本 (Normal Only)，用于无监督/自监督学习。
    2. 测试集包含剩余的正常样本和所有异常样本 (Anomaly)。

    Args:
        df (pd.DataFrame): 包含完整数据的 DataFrame。
        output_dir (str): 输出目录。
        train_size (int): 训练集大小。默认 4855 是 LogBERT 论文使用的 HDFS 基准值。
        random_state (int): 随机种子，保证结果可复现。

    Returns:
        无 (结果直接写入文件)
    """
    print("[-] Splitting Train/Test...")

    # 布尔索引：筛选正常和异常样本
    # df_dataset shape 变化: (N, 3) -> (N_normal, 3) 和 (N_abnormal, 3)
    normal_df = df[df['Label'] == 0]
    abnormal_df = df[df['Label'] == 1]

    # 洗牌正常数据
    # sample(frac=1) 表示抽取 100% 的数据，相当于随机打乱
    # reset_index(drop=True) 重置索引，避免索引混乱
    normal_df = normal_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 执行切片划分
    # 训练集：前 train_size 个正常样本
    train_df = normal_df.iloc[:train_size]

    # 测试集 (Normal)：剩余的正常样本
    test_normal_df = normal_df.iloc[train_size:]

    # 测试集 (Anomaly)：所有的异常样本
    test_abnormal_df = abnormal_df

    # 打印统计信息
    print(f"    Train (Normal): {len(train_df)}")
    print(f"    Test (Normal) : {len(test_normal_df)}")
    print(f"    Test (Anomaly): {len(test_abnormal_df)}")

    # 保存 CSV 格式 (推荐格式，包含 Metadata)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_normal_df.to_csv(os.path.join(output_dir, "test_normal.csv"), index=False)
    test_abnormal_df.to_csv(os.path.join(output_dir, "test_abnormal.csv"), index=False)

    # 保存纯文本格式 (兼容性保存)
    # 某些旧的数据加载器 (Dataset) 可能只读取纯文本行
    # zip() 将文件名列表和 DataFrame 列表打包，便于循环处理
    for name, data in zip(['train', 'test_normal', 'test_abnormal'],
                          [train_df, test_normal_df, test_abnormal_df]):
        file_path = os.path.join(output_dir, f"{name}.txt")
        with open(file_path, 'w') as f:
            for seq in data['EventSequence']:
                # 写入每一行序列，并手动添加换行符
                f.write(seq + "\n")

    print("[-] Data split completed.")


# ==================== 4. 程序主入口 ====================

if __name__ == "__main__":
    # 使用 os.path.join 拼接路径，兼容 Windows/Linux 系统
    log_path = os.path.join(INPUT_DIR, LOG_FILE)
    label_path = os.path.join(INPUT_DIR, LABEL_FILE)

    # 简单的文件存在性检查
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        # 退出码 1 表示异常退出
        sys.exit(1)

    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        sys.exit(1)

    # 步骤 1: 执行清洗和预处理
    # 返回处理后的 DataFrame
    df_result = hdfs_preprocess_no_parser(log_path, label_path, OUTPUT_DIR)

    # 步骤 2: 划分数据集
    # n=4855 是 LogBERT 论文复现 HDFS 实验时的标准基线设定
    split_train_test(df_result, OUTPUT_DIR, train_size=4855)