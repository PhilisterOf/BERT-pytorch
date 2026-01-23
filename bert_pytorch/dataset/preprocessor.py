import sys
import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ================= 配置路径 =================
# 建议输入目录改为存放 HDFS.log 的原始路径
input_dir = '../../data/raw/'
output_dir = '../../data/processed/'  # 新的输出目录
log_file = "HDFSS.log"
label_file = "anomaly_label.csv"

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ================= 1. 定义 Parser-Free 清洗器 =================
class ParserFreeTokenizer:
    def __init__(self):
        self.patterns = [
            # 1. 优先处理 Block ID：保留 blk_ 前缀，将后面的数字变成 <NUM>
            # 原始: blk_-1608999687919862906 -> 清洗后: blk_ <NUM>
            (r'(?<=blk_)[-\d]+', ' <NUM> '),

            # 2. IP 地址 -> <IP>
            (r'(\d{1,3}\.){3}\d{1,3}', ' <IP> '),

            # 3. 路径/文件 (HDFS中常见) -> <PATH>
            # 匹配以 / 开头包含字母数字的串
            (r'(/[-\w\.]+)+', ' <PATH> '),

            # 4. 十六进制 (常见于某些内存地址或校验码) -> <HEX>
            (r'0x[0-9a-fA-F]+', ' <HEX> '),

            # 5. 纯数字 (兜底) -> <NUM>
            (r'\d+', ' <NUM> ')
        ]

    def tokenize(self, text):
        # 转小写
        text = text.lower()

        # 应用所有正则替换
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)

        # 去除多余的非字母数字符号 (保留 < > _ 用于 token)
        # 这一步是为了让 BERT 的 tokenizer 更干净，也可以选择保留标点
        text = re.sub(r'[^\w\s<>]', ' ', text)

        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ================= 2. 核心处理逻辑 (替代原来的 hdfs_sampling) =================
def process_hdfs_data(log_file_path, label_file_path):
    print(f"Processing raw log: {log_file_path}")

    # 初始化清洗器
    tokenizer = ParserFreeTokenizer()

    # 用于存储分组后的日志: {block_id: [clean_log1, clean_log2, ...]}
    data_dict = defaultdict(list)

    # 预编译正则用于提取 Block ID (这是分组的关键，绝对不能丢)
    blk_pattern = re.compile(r'(blk_-?\d+)')

    # 逐行读取原始日志 (不使用 Pandas 读取 Raw Text，防止格式错误，且更快)
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Scanning Logs"):
            # A. 提取 Block ID (用于分组)
            match = blk_pattern.search(line)
            if not match:
                continue  # 如果没有 Block ID，这条日志归属不明，跳过

            block_id = match.group(1)

            # B. 提取日志内容 (去除 HDFS 头部的时间戳等噪声)
            # HDFS 格式通常是: Date Time Pid Level Component: Content
            # 简单的策略是：取第一个冒号后面的内容，或者保留 Component + Content
            # 示例: 081109 203615 148 INFO dfs.DataNode$PacketResponder: ...

            # 策略：找到第一个 'INFO', 'WARN', 'ERROR' 等 Level 之后的内容
            # 或者简单粗暴：直接清洗整行，依靠 BERT 忽略前面的数字日期
            # 这里推荐：简单清洗整行，但去掉前两个 Token (Date, Time)

            # --- 清洗逻辑 ---
            clean_text = tokenizer.tokenize(line)

            # 存入字典
            data_dict[block_id].append(clean_text)

    print(f"Total blocks extracted: {len(data_dict)}")

    # 加载标签
    print(f"Loading labels from {label_file_path}")
    label_df = pd.read_csv(label_file_path)
    # 转成字典: {blk_xxxx: 1 (异常), blk_yyyy: 0 (正常)}
    blk_label_dict = dict(zip(label_df['BlockId'], label_df['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)))

    # 构建最终数据集列表
    # 格式: [{'BlockId': xxx, 'EventSequence': ['log1', 'log2'], 'Label': 0}, ...]
    dataset = []
    for blk_id, logs in data_dict.items():
        # 如果标签文件中没有这个 block，通常设为正常或丢弃，这里假设为正常(0)或跳过
        if blk_id in blk_label_dict:
            dataset.append({
                'BlockId': blk_id,
                'EventSequence': logs,  # 这里存的是文本列表，不再是 ID 列表
                'Label': blk_label_dict[blk_id]
            })

    return pd.DataFrame(dataset)


# ================= 3. 数据集划分与保存 =================
def save_parser_free_dataset(df, output_dir, train_ratio=0.8):
    # 分离正常和异常
    normal_df = df[df['Label'] == 0]
    abnormal_df = df[df['Label'] == 1]

    # 划分训练集和测试集 (仅用正常数据训练 BERT)
    # 注意：BERT 预训练通常只需要正常数据，或者把异常数据当作测试集

    # 这里的逻辑参考原本代码：按比例划分正常数据
    train_normal, test_normal = train_test_split(normal_df, train_size=train_ratio, random_state=42)

    print(f"Train Normal: {len(train_normal)}")
    print(f"Test Normal: {len(test_normal)}")
    print(f"Test Abnormal: {len(abnormal_df)}")

    # 保存为 JSONL 格式 (JSON Lines)
    # 相比 txt，JSONL 更适合存储列表结构的日志序列，读写方便

    def save_jsonl(dataframe, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for _, row in dataframe.iterrows():
                # 我们只需要保存序列内容即可
                line_obj = {
                    "BlockId": row['BlockId'],
                    "Label": int(row['Label']),
                    "EventSequence": row['EventSequence']  # list of strings
                }
                f.write(json.dumps(line_obj) + '\n')

    save_jsonl(train_normal, "train.json")
    save_jsonl(test_normal, "test_normal.json")
    save_jsonl(abnormal_df, "test_abnormal.json")
    print(f"All data saved to {output_dir}")


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 组合完整路径
    log_path = os.path.join(input_dir, log_file)
    label_path = os.path.join(input_dir, label_file)

    # 2. 处理数据 (Group & Normalize)
    full_df = process_hdfs_data(log_path, label_path)

    # 3. 保存
    # n=4855 是原论文的设置，这里我们使用比例划分，更通用
    save_parser_free_dataset(full_df, output_dir, train_ratio=0.8)