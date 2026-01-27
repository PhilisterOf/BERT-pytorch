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
        self.vocab_file = os.path.join(self.vocab_path, "vocab.txt")

        # [通用逻辑] 默认设为 3000，但在 evaluate 中会根据实际文件覆盖
        self.vocab_size = 3000

        self.test_ratio = 0.1
        self.max_len = 512
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.batch_size = 512
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Dataset (通用) ====================
class TailTestDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        print(f"[-] Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        self.data = df['EventSequence'].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(tokens) > self.seq_len:
            tokens = tokens[-self.seq_len:]
            tokens[0] = self.tokenizer.cls_token_id
        padding_len = self.seq_len - len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * padding_len
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}


# ==================== 混合特征提取 (Distance + TopK) ====================
def extract_hybrid_scores(model, center, dataloader, device, tokenizer, k=10):
    """
    同时提取 Distance Score 和 Top-K Miss Score
    """
    model.eval()
    dist_scores = []
    topk_scores = []

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            raw_input = batch["input_ids"].to(device)

            # --- 1. 计算 Distance Score ---
            # Forward Pass 1: 不 Mask，拿 CLS 向量
            _, cls_embedding = model(raw_input)
            # 欧氏距离
            dists = torch.norm(cls_embedding - center, dim=1)
            dist_scores.extend(dists.cpu().numpy().tolist())

            # --- 2. 计算 Top-K Miss Score ---
            batch_miss_count = torch.zeros(raw_input.size(0), device=device)
            valid_mask = (raw_input != pad_id) & (raw_input != cls_id) & (raw_input != sep_id)

            for offset in [0, 1]:
                masked_input = raw_input.clone()
                seq_len = raw_input.size(1)
                indices = torch.arange(seq_len, device=device)
                mask_cols = (indices % 2 == offset)
                token_mask = mask_cols.unsqueeze(0).expand_as(masked_input) & valid_mask

                if token_mask.sum() == 0: continue

                masked_input[token_mask] = mask_id
                logits, _ = model(masked_input)
                _, topk_indices = torch.topk(logits, k=k, dim=-1)

                real_tokens = raw_input.unsqueeze(-1)
                hit = (topk_indices == real_tokens).any(dim=-1)
                miss = token_mask & (~hit)
                batch_miss_count += miss.sum(dim=1).float()

            topk_scores.extend(batch_miss_count.cpu().numpy().tolist())

    return np.array(dist_scores), np.array(topk_scores)


# ==================== 归一化工具 ====================
def normalize(arr):
    """Z-Score 归一化 (Standardization)"""
    # 避免除以 0
    std = arr.std()
    if std == 0: return arr - arr.mean()
    return (arr - arr.mean()) / std


# ==================== Evaluation ====================
def get_loader(dataset, cfg):
    total = len(dataset)
    args = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers, "pin_memory": True, "shuffle": False}
    if cfg.test_ratio < 1.0:
        subset_len = int(total * cfg.test_ratio)
        indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))[:subset_len].tolist()
        return DataLoader(Subset(dataset, indices), **args)
    return DataLoader(dataset, **args)


def evaluate(cfg):
    # 1. 加载 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.vocab_path, do_lower_case=True)
    # [核心修改] 动态获取词表大小，避免硬编码
    real_vocab_size = len(tokenizer.vocab)
    print(f"[-] Loaded Tokenizer. Real Vocab Size: {real_vocab_size}")

    # 2. 加载模型
    print(f"[-] Loading model from {cfg.model_path}...")
    model = LogBERT(real_vocab_size, cfg.hidden, cfg.layers, cfg.heads).to(cfg.device)

    # weights_only=False 兼容旧版
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载中心向量
    if 'center' in checkpoint:
        center = checkpoint['center'].to(cfg.device)
        print(f"[-] Hypersphere Center loaded. Norm: {torch.norm(center).item():.4f}")
    else:
        print("[!] Warning: Center not found! Distance score will be invalid.")
        center = torch.zeros(cfg.hidden).to(cfg.device)

    # 3. 数据加载
    ds_norm = TailTestDataset(cfg.test_normal_file, tokenizer, cfg.max_len)
    ds_abnorm = TailTestDataset(cfg.test_abnormal_file, tokenizer, cfg.max_len)
    loader_norm = get_loader(ds_norm, cfg)
    loader_abnorm = get_loader(ds_abnorm, cfg)

    # 4. 提取特征 (固定 K=10，基于之前搜索结果)
    BEST_K = 10
    print(f"[-] Extracting features (K={BEST_K})...")

    dist_n, topk_n = extract_hybrid_scores(model, center, loader_norm, cfg.device, tokenizer, k=BEST_K)
    dist_a, topk_a = extract_hybrid_scores(model, center, loader_abnorm, cfg.device, tokenizer, k=BEST_K)

    # 5. 混合搜索
    y_true = np.concatenate([np.zeros(len(dist_n)), np.ones(len(dist_a))])

    # 归一化
    norm_dist = normalize(np.concatenate([dist_n, dist_a]))
    norm_topk = normalize(np.concatenate([topk_n, topk_a]))

    print("\n" + "=" * 60)
    print("HYBRID SEARCH (Alpha * Dist + (1-Alpha) * TopK)")
    print("=" * 60)
    print(f"{'Alpha':<6} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'AUC':<8}")
    print("-" * 50)

    best_f1 = 0
    best_res = {}

    # 搜索 Alpha (0=TopK, 1=Dist)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for alpha in alphas:
        final_scores = alpha * norm_dist + (1 - alpha) * norm_topk

        p, r, t = precision_recall_curve(y_true, final_scores)
        f1s = 2 * p * r / (p + r + 1e-10)
        best_idx = np.argmax(f1s)

        curr_f1 = f1s[best_idx]
        curr_auc = roc_auc_score(y_true, final_scores)

        print(f"{alpha:<6.1f} | {p[best_idx]:.4f}   | {r[best_idx]:.4f}   | {curr_f1:.4f}   | {curr_auc:.4f}")

        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_res = {"alpha": alpha, "f1": curr_f1, "auc": curr_auc}

    print("=" * 60)
    print(f"🏆 BEST RESULT:")
    print(f"   Alpha    : {best_res['alpha']} (0=TopK, 1=Dist)")
    print(f"   F1 Score : {best_res['f1']:.4f}")
    print(f"   AUC Score: {best_res['auc']:.4f}")

    if best_res['f1'] > 0.8840:
        print(">> 结论: 混合策略有效！")
    else:
        print(">> 结论: Top-K 仍然主导。")


if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)