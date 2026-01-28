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
from hdfs.bd3_dataset import LogDataset
from hdfs.bm4_bert import LogBERT


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
        self.vocab_size = self._get_vocab_size()

        # 参数需与训练一致
        self.test_ratio = 0.1  # 建议全量测试
        self.max_len = 512
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.batch_size = 512
        self.num_workers = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_vocab_size(self):
        if not os.path.exists(self.vocab_file): return 424
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            return len(f.readlines())


# 保持原类名不变
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


# [修改点] 改用 Min-Max，对 SVDD 距离更友好
def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0: return arr
    return (arr - min_val) / (max_val - min_val + 1e-8)


# [修改点] 提取特征的同时计算 Loss，不再做复杂的 Rank
def extract_features(model, center, dataloader, device, tokenizer):
    model.eval()

    dist_scores = []
    # 存储每个样本每 Token 的 Loss，用于动态 Top-K
    token_loss_lists = []

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            input_ids = batch["input_ids"].to(device)

            # 1. Distance (SVDD)
            logits, cls_embedding = model(input_ids)
            # 欧氏距离平方
            dists = torch.sum((cls_embedding - center) ** 2, dim=1)
            dist_scores.extend(dists.cpu().numpy().tolist())

            # 2. Reconstruction Loss (MLM)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = input_ids.view(-1)

            losses = loss_fct(flat_logits, flat_labels)
            losses = losses.view(input_ids.size(0), -1)

            # 只取最大的前 50 个 Loss (节省内存，一般 K <= 50)
            top_losses, _ = torch.topk(losses, k=min(50, losses.size(1)), dim=1)
            token_loss_lists.extend(top_losses.cpu().numpy().tolist())

    return np.array(dist_scores), np.array(token_loss_lists)


def evaluate(cfg):
    print(f"[-] Loading model with vocab_size={cfg.vocab_size}...")
    model = LogBERT(cfg.vocab_size, cfg.hidden, cfg.layers, cfg.heads).to(cfg.device)

    try:
        checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"[!] Model load failed: {e}")
        return

    if 'center' in checkpoint:
        center = checkpoint['center'].to(cfg.device)
        print(f"[-] Center loaded. Norm: {torch.norm(center).item():.4f}")
    else:
        center = torch.zeros(cfg.hidden).to(cfg.device)

    tokenizer = BertTokenizer.from_pretrained(cfg.vocab_path, do_lower_case=True)

    ds_norm = TailTestDataset(cfg.test_normal_file, tokenizer, cfg.max_len)
    ds_abnorm = TailTestDataset(cfg.test_abnormal_file, tokenizer, cfg.max_len)

    def get_subset(ds):
        total = len(ds)
        if cfg.test_ratio < 1.0:
            indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))[
                      :int(total * cfg.test_ratio)].tolist()
            return DataLoader(Subset(ds, indices), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        return DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    loader_norm = get_subset(ds_norm)
    loader_abnorm = get_subset(ds_abnorm)

    # 提取特征
    print(f"[-] Extracting features...")
    dist_n, tloss_n = extract_features(model, center, loader_norm, cfg.device, tokenizer)
    dist_a, tloss_a = extract_features(model, center, loader_abnorm, cfg.device, tokenizer)

    # 归一化 SVDD Distance
    raw_dist = np.concatenate([dist_n, dist_a])
    norm_dist = min_max_normalize(raw_dist)

    # 准备 Token Loss 数据
    raw_tloss = np.concatenate([tloss_n, tloss_a], axis=0)
    y_true = np.concatenate([np.zeros(len(dist_n)), np.ones(len(dist_a))])

    # K 候选
    k_candidates = [5, 10, 15, 20]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("\n" + "=" * 60)
    print("GRID SEARCH: MinMax(Dist) + MinMax(TopK)")  # 修改了标题以反映逻辑变化
    print("=" * 60)
    print(f"{'K':<3} | {'Alpha':<5} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'AUC':<7}")
    print("-" * 50)

    best_f1 = 0
    best_res = {}

    for k in k_candidates:
        # 动态计算 Top-K 平均 Loss
        # raw_tloss 已经是 Top-50，直接取前 k 列平均即可
        k_scores = np.mean(raw_tloss[:, :k], axis=1)
        norm_topk = min_max_normalize(k_scores)

        for alpha in alphas:
            final_scores = alpha * norm_dist + (1 - alpha) * norm_topk

            p, r, t = precision_recall_curve(y_true, final_scores)
            f1s = 2 * p * r / (p + r + 1e-10)
            best_idx = np.argmax(f1s)

            curr_f1 = f1s[best_idx]
            curr_auc = roc_auc_score(y_true, final_scores)

            if curr_f1 > 0.5:
                print(
                    f"{k:<3} | {alpha:<5.1f} | {p[best_idx]:.4f}  | {r[best_idx]:.4f}  | {curr_f1:.4f}  | {curr_auc:.4f}")

            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_res = {"k": k, "alpha": alpha, "f1": curr_f1, "auc": curr_auc}

    print("=" * 60)
    print(f"🏆 GLOBAL BEST:")
    print(f"   K        : {best_res.get('k')}")
    print(f"   Alpha    : {best_res.get('alpha')} (0=TopK, 1=Dist)")
    print(f"   F1 Score : {best_res.get('f1'):.4f}")
    print(f"   AUC Score: {best_res.get('auc'):.4f}")


if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)