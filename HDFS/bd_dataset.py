import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer


class LogDataset(Dataset):
    """
    LogBERT 数据集加载类 (Parser-Free 版本)

    功能：
    1. 读取清洗后的 CSV 日志数据。
    2. 使用 BertTokenizer 将文本转换为 ID 序列。
    3. 执行 MLM (Masked Language Modeling) 的随机遮盖策略。
    4. [创新点] 执行 Random Shuffle 构造对比学习的负样本。
    """

    def __init__(self, data_path, vocab_path, seq_len=128, min_len=5, encoding="utf-8"):
        """
        Args:
            data_path (str): CSV 数据文件路径 (e.g., train.csv)
            vocab_path (str): vocab.txt 所在的目录路径
            seq_len (int): 序列最大长度 (截断/填充)
            min_len (int): 过滤掉太短的序列
        """
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.min_len = min_len

        # 1. 加载 Tokenizer (替代旧代码中的 WordVocab)
        # do_lower_case=True 对应我们在 train_tokenizer 中的 lowercase=True
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)

        # 2. 读取数据
        print(f"[-] Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)

        # 过滤空数据
        self.df['EventSequence'] = self.df['EventSequence'].fillna("")
        self.total_lines = len(self.df)
        print(f"    Loaded {self.total_lines} samples.")

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        # 获取原始文本 (Parser-Free Text)
        text = self.df.iloc[idx]['EventSequence']
        label = self.df.iloc[idx].get('Label', 0)  # 默认为 0 (Normal)

        # ================= Step 1: Tokenization =================
        # 使用 HuggingFace Tokenizer 转换文本为 ID
        # add_special_tokens=True 会自动添加 [CLS] 和 [SEP]
        # 注意：这里我们不立即做 padding，因为要先做 mask 和 shuffle
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.seq_len)

        # 给 tokens 起了一个别名（Alias），叫 input_ids
        input_ids = tokens

        # 长度校验
        if len(input_ids) < self.min_len:
            # 如果太短，为了工程健壮性，我们补齐到一个最小长度或直接返回 Padding
            # 这里选择补齐并标记为不参与 loss 计算
            padding = [self.tokenizer.pad_token_id] * (self.seq_len - len(input_ids))
            return {
                "bert_input": torch.tensor(input_ids + padding),
                # TODO: 全量 mask，确保 CrossEntropyLoss 完全忽略此样本
                # "bert_label": torch.tensor([-100] * self.seq_len),
                "bert_label": torch.tensor([-100] * len(input_ids) + padding),
                "contrastive_pos": torch.tensor(input_ids + padding),
                "contrastive_neg": torch.tensor(input_ids + padding),  # 太短无法 shuffle
                "segment_label": torch.tensor(label)
            }

        # ================= Step 2: Masked Language Model (MLM) =================
        # 参考原 bert_pytorch 的 random_word 逻辑
        # bert_input: 被挖空的序列
        # bert_label: 原始值的标签 (非 Mask 位置为 -100)
        bert_input, bert_label = self.random_word(input_ids)

        # ================= Step 3: Contrastive Learning Augmentation =================
        # [创新点] 构造 "Hard Negative" (困难负样本)
        # 逻辑：将原始序列随机打乱 (Shuffle)，破坏时序逻辑
        # 原始序列 (Anchor) -> contrastive_pos
        # 乱序序列 (Negative) -> contrastive_neg
        contrastive_pos = input_ids[:]  # 浅拷贝
        contrastive_neg = self.random_shuffle(input_ids[:], self.min_len)

        # ================= Step 4: Padding =================
        # 统一补齐到 seq_len
        padding_len = self.seq_len - len(input_ids)

        # 辅助函数: 执行 padding
        def pad(seq, val=0):
            return seq + [val] * padding_len

        # [PAD] ID 通常为 0
        pad_id = self.tokenizer.pad_token_id

        output = {
            "bert_input": torch.tensor(pad(bert_input, pad_id)),
            "bert_label": torch.tensor(pad(bert_label, -100)),  # Loss 计算忽略 -100

            # 对比学习用的正负样本 (不 Mask，只 Pad)
            "contrastive_pos": torch.tensor(pad(contrastive_pos, pad_id)),
            "contrastive_neg": torch.tensor(pad(contrastive_neg, pad_id)),

            "segment_label": torch.tensor(label, dtype=torch.long)  # 0=Normal, 1=Anomaly
        }

        return output

    def random_word(self, sentence):
        """
        实现 BERT 的 Masking 策略 (参考原 bert_pytorch 逻辑)
        Args:
            sentence: list of token ids
        Returns:
            output_label: 只有被 Mask 的位置有值，其余为 -100
        """
        # 浅拷贝 (Shallow Copy)（切片操作）。Python 会开辟一个新的内存房间，把数据复制过去给 tokens
        tokens = sentence[:]
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()

            # 跳过特殊 Token ([CLS], [SEP])
            # HuggingFace 中通常 101=[CLS], 102=[SEP] (具体看vocab)
            # 更通用的写法是判断是否在该集合内
            if token in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                output_label.append(-100)
                continue

            # 15% 的概率进行 Mask
            if prob < 0.15:
                # 80% 几率 -> [MASK]
                # 10% 几率 -> 随机词
                # 10% 几率 -> 保持原样 (但 label 依然记录，用于让模型学习"这是对的")

                prob /= 0.15    # 放大！现在 prob 是 0~1 之间了

                if prob < 0.8:  # 直观！就是 80%
                    tokens[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:    # 直观！就是 80%~90% 之间 (即 10%)
                    # 随机选一个词 (从词表中随机)
                    tokens[i] = random.randrange(len(self.tokenizer.vocab))
                else:   # 直观！就是剩下的 10%
                    # 保持原样
                    pass

                output_label.append(token)  # 记录真实值用于计算 Loss
            else:
                output_label.append(-100)  # 不计算 Loss

        return tokens, output_label

    def random_shuffle(self, sentence, min_len):
        """
        [创新点] 随机打乱序列，作为对比学习的负样本。
        只打乱中间的内容，保留 [CLS] 和 [SEP]
        """
        # 找到 [CLS] 和 [SEP] 的位置
        # 通常 sentence[0] 是 CLS, sentence[-1] 是 SEP
        if len(sentence) <= min_len:
            return sentence  # 太短不打乱

        # 提取中间部分
        middle = sentence[1:-1]
        random.shuffle(middle)

        # 重新拼接
        return [sentence[0]] + middle + [sentence[-1]]


# ==================== 单元测试 ====================
if __name__ == "__main__":
    # 简单的测试代码，确保 dataset 能跑通
    vocab_dir = "../output/hdfs/"  # 你的 vocab.txt 目录
    data_file = "../output/hdfs/train.csv"  # 你的训练数据

    # 假设文件存在，尝试初始化
    import os

    if os.path.exists(vocab_dir) and os.path.exists(data_file):
        ds = LogDataset(data_file, vocab_dir)
        print("Sample 0:", ds[0])
        print("BERT Input Shape:", ds[0]['bert_input'].shape)
        print("Contrastive Neg (Shuffled):", ds[0]['contrastive_neg'])
    else:
        print("Skipping test: files not found.")