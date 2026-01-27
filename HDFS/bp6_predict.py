import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import random
from HDFS.bd3_dataset import LogDataset
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

        self.test_ratio = 0.1
        self.max_len = 512
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.0
        self.batch_size = 512
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 尾部截断 Dataset ====================
class TailTestDataset(Dataset):
    def __init__(self, file_path, vocab_path, seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        self.seq_len = seq_len
        print(f"[-] Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        self.data = df['EventSequence'].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # 不截断编码
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)

        # [核心] 如果超长，取最后 512 个
        if len(tokens) > self.seq_len:
            tokens = tokens[-self.seq_len:]
            # 修正第一个词为 CLS (可选，但推荐)
            tokens[0] = self.tokenizer.cls_token_id

        # Padding
        padding_len = self.seq_len - len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * padding_len
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}


# ==================== Top-K 推理 ====================
def compute_topk_scores(model, dataloader, device, tokenizer, k=10):
    model.eval()
    scores = []

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference (Top-{k})"):
            raw_input = batch["input_ids"].to(device)
            batch_scores = torch.zeros(raw_input.size(0), device=device)

            valid_mask = (raw_input != pad_id) & (raw_input != cls_id) & (raw_input != sep_id)

            for offset in [0, 1]:
                masked_input = raw_input.clone()
                seq_len = raw_input.size(1)
                indices = torch.arange(seq_len, device=device)
                mask_cols = (indices % 2 == offset)
                token_mask = mask_cols.unsqueeze(0).expand_as(masked_input) & valid_mask

                masked_input[token_mask] = mask_id

                logits, _ = model(masked_input)
                _, topk_indices = torch.topk(logits, k=k, dim=-1)

                real_tokens = raw_input.unsqueeze(-1)
                hit = (topk_indices == real_tokens).any(dim=-1)

                # 没命中 TopK 算异常
                miss = token_mask & (~hit)
                batch_scores += miss.sum(dim=1).float()

            scores.extend(batch_scores.cpu().numpy().tolist())
    return np.array(scores)


# ==================== Evaluation ====================
def get_loader(dataset, cfg):
    total = len(dataset)
    args = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers, "pin_memory": True, "shuffle": False}
    if cfg.test_ratio < 1.0:
        subset_len = int(total * cfg.test_ratio)
        indices = torch.randperm(total)[:subset_len].tolist()
        return DataLoader(Subset(dataset, indices), **args)
    return DataLoader(dataset, **args)


def evaluate(cfg):
    print(f"[-] Loading model...")
    dummy_tokenizer = BertTokenizer.from_pretrained(cfg.vocab_path, do_lower_case=True)
    vocab_size = len(dummy_tokenizer.vocab)
    model = LogBERT(vocab_size, cfg.hidden, cfg.layers, cfg.heads).to(cfg.device)
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device)  # , weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    ds_norm = TailTestDataset(cfg.test_normal_file, cfg.vocab_path, cfg.max_len)
    ds_abnorm = TailTestDataset(cfg.test_abnormal_file, cfg.vocab_path, cfg.max_len)

    loader_norm = get_loader(ds_norm, cfg)
    loader_abnorm = get_loader(ds_abnorm, cfg)

    # 构造标签
    # 注意：如果用了采样，这里长度要动态获取
    # 为简单起见，我们直接拼结果

    print("\n" + "=" * 60)
    print(f"FINAL TOP-K EVALUATION (Ratio={cfg.test_ratio})")
    print("=" * 60)

    for k in [5, 10]:
        print(f"[-] Scoring Top-{k}...")
        n_scores = compute_topk_scores(model, loader_norm, cfg.device, dummy_tokenizer, k=k)
        a_scores = compute_topk_scores(model, loader_abnorm, cfg.device, dummy_tokenizer, k=k)

        y_true = [0] * len(n_scores) + [1] * len(a_scores)
        y_scores = np.concatenate([n_scores, a_scores])

        p, r, t = precision_recall_curve(y_true, y_scores)
        f1 = 2 * p * r / (p + r + 1e-10)
        best_idx = np.argmax(f1)

        print(f"Strategy: Top-{k:<2} | Best F1: {f1[best_idx]:.4f} | AUC: {roc_auc_score(y_true, y_scores):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)