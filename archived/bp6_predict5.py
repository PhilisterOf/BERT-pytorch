import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import random
from HDFS.bd3_dataset import LogDataset
from torch.utils.data import DataLoader, Dataset, Subset # <--- 加上 Subset
from HDFS.bm4_bert import LogBERT


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config:
    def __init__(self):
        self.output_dir = "../output/hdfs/"
        self.model_path = "../output/hdfs/best_model.pth"
        self.test_normal_file = "../output/hdfs/test_normal.csv"
        self.test_abnormal_file = "../output/hdfs/test_abnormal.csv"
        self.vocab_path = "../output/hdfs/"

        # TODO: 修改测试集使用比例
        self.test_ratio = 0.1
        self.max_len = 512  # 视野开大，尽量少截断
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.0

        # [极速模式] Batch Size 开大
        self.batch_size = 512
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 极简 Dataset (无滑动窗口) ====================
class SimpleTestDataset(Dataset):
    def __init__(self, file_path, vocab_path, seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        self.seq_len = seq_len

        print(f"[-] Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        # 读取所有文本
        self.data = df['EventSequence'].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # [核心] 直接截断 (Truncation=True)
        # 超过 512 的部分直接丢弃，只看前 512
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.seq_len
        )

        # Padding
        padding_len = self.seq_len - len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long)
        }


# ==================== 核心推理逻辑 ====================
def compute_scores_fast(model, dataloader, device, tokenizer):
    model.eval()
    criterion_mlm = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    final_scores = []

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    # [新增] 获取 CLS 和 SEP 的 ID
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            raw_input = batch["input_ids"].to(device)

            batch_max_losses = torch.zeros(raw_input.size(0), device=device)

            # [核心修复]
            # 排除 [SEP] 和 [CLS] 参与评分！
            # 否则它们的高 Loss 会掩盖真正的异常词
            valid_mask = (raw_input != pad_id) & (raw_input != cls_id) & (raw_input != sep_id)

            for offset in [0, 1]:
                masked_input = raw_input.clone()
                labels = raw_input.clone()

                seq_len = raw_input.size(1)
                indices = torch.arange(seq_len, device=device)
                mask_cols = (indices % 2 == offset)

                token_mask = mask_cols.unsqueeze(0).expand_as(masked_input) & valid_mask

                masked_input[token_mask] = mask_id
                labels[~token_mask] = -100

                logits, _ = model(masked_input)

                vocab_size = logits.size(-1)
                loss_per_token = criterion_mlm(logits.view(-1, vocab_size), labels.view(-1))
                loss_per_token = loss_per_token.view(raw_input.size(0), -1)

                # 只在 valid_mask 为 True 的地方取值
                current_max, _ = (loss_per_token * token_mask.float()).max(dim=1)
                batch_max_losses = torch.max(batch_max_losses, current_max)

            final_scores.extend(batch_max_losses.cpu().numpy().tolist())

    return np.array(final_scores)


# ==================== 新增：带采样功能的 Loader ====================
def get_loader(dataset, cfg, desc="Dataset"):
    total_len = len(dataset)

    # 通用参数
    loader_args = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": True
    }

    if cfg.test_ratio < 1.0:
        # 计算采样数量
        subset_len = int(total_len * cfg.test_ratio)
        subset_len = max(1, subset_len)
        print(f"[-] {desc} Sampling: {cfg.test_ratio * 100}% ({subset_len}/{total_len})")

        # 随机采样
        indices = torch.randperm(total_len)[:subset_len].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(subset, **loader_args)
    else:
        # 全量
        print(f"[-] {desc} Full Usage: {total_len} samples")
        return DataLoader(dataset, **loader_args)


# ==================== 修正后的 evaluate 函数 ====================
def evaluate(cfg):
    print(f"[-] Loading model from {cfg.model_path}")
    dummy_tokenizer = BertTokenizer.from_pretrained(cfg.vocab_path, do_lower_case=True)
    vocab_size = len(dummy_tokenizer.vocab)

    model = LogBERT(vocab_size=vocab_size, hidden=cfg.hidden, n_layers=cfg.layers, attn_heads=cfg.heads).to(cfg.device)

    # 加载权重
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 1. 准备 Dataset
    ds_normal = SimpleTestDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len)
    ds_abnormal = SimpleTestDataset(cfg.test_abnormal_file, cfg.vocab_path, seq_len=cfg.max_len)

    # 2. 获取 Loader (这里修复了：调用 get_loader 进行采样)
    loader_normal = get_loader(ds_normal, cfg, "Normal")
    loader_abnormal = get_loader(ds_abnormal, cfg, "Abnormal")

    # 3. 计算分数
    print("[-] Scoring Normal Samples...")
    n_scores = compute_scores_fast(model, loader_normal, cfg.device, dummy_tokenizer)

    print("[-] Scoring Abnormal Samples...")
    a_scores = compute_scores_fast(model, loader_abnormal, cfg.device, dummy_tokenizer)

    # 4. 评估
    y_true = [0] * len(n_scores) + [1] * len(a_scores)
    y_scores = np.concatenate([n_scores, a_scores])

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)

    print("\n" + "=" * 60)
    print("FINAL FAST EVALUATION (Truncated & Masked SEP)")
    print("=" * 60)
    print(f"Best F1: {f1_scores[best_idx]:.4f}")
    print(f"AUC:     {roc_auc_score(y_true, y_scores):.4f}")
    print(f"Normal Mean:   {n_scores.mean():.4f}")
    print(f"Abnormal Mean: {a_scores.mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)
    else:
        print("Model not found.")