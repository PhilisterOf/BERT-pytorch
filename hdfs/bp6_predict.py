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

        # 自动检测词表大小
        self.vocab_size = self._get_vocab_size()

        # 注意：这里必须与你【训练时】的参数保持一致！
        # 如果你还没重训，就保持 128/2/4
        self.test_ratio = 0.1
        self.max_len = 512
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.batch_size = 512
        self.num_workers = 0  # Windows下设为0最稳
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_vocab_size(self):
        if not os.path.exists(self.vocab_file): return 424
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            return len(f.readlines())


# ==================== Tail Truncation Dataset ====================
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


# ==================== Z-Score 归一化 (核心改进) ====================
def z_score_normalize(arr):
    """
    使用 Z-Score (标准化) 而非 Min-Max。
    抵抗 Distance 中的离群值干扰。
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0: return arr - mean
    return (arr - mean) / std


# ==================== 特征提取 ====================
def extract_features(model, center, dataloader, device, tokenizer, k_candidates):
    model.eval()

    # 存储特征
    dist_scores = []
    # 存储不同 K 下的 Miss Count: {k: [score...]}
    topk_scores_map = {k: [] for k in k_candidates}

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            raw_input = batch["input_ids"].to(device)

            # 1. Distance
            _, cls_embedding = model(raw_input)
            dists = torch.norm(cls_embedding - center, dim=1)
            dist_scores.extend(dists.cpu().numpy().tolist())

            # 2. Multi-K TopK Miss
            batch_size, seq_len = raw_input.size()
            batch_ranks_map = {k: torch.zeros(batch_size, device=device) for k in k_candidates}

            valid_mask = (raw_input != pad_id) & (raw_input != cls_id) & (raw_input != sep_id)

            for offset in [0, 1]:
                masked_input = raw_input.clone()
                indices = torch.arange(seq_len, device=device)
                mask_cols = (indices % 2 == offset)
                token_mask = mask_cols.unsqueeze(0).expand_as(masked_input) & valid_mask

                if token_mask.sum() == 0: continue

                masked_input[token_mask] = mask_id
                logits, _ = model(masked_input)  # [B, L, V]

                # 计算 Rank
                real_tokens = raw_input
                target_scores = logits.gather(-1, real_tokens.unsqueeze(-1)).squeeze(-1)
                ranks = (logits > target_scores.unsqueeze(-1)).sum(dim=-1) + 1  # 1-based rank

                for k in k_candidates:
                    miss = (ranks > k) & token_mask
                    batch_ranks_map[k] += miss.sum(dim=1).float()

            for k in k_candidates:
                topk_scores_map[k].extend(batch_ranks_map[k].cpu().numpy().tolist())

    # 转 Numpy
    dist_arr = np.array(dist_scores)
    for k in k_candidates:
        topk_scores_map[k] = np.array(topk_scores_map[k])

    return dist_arr, topk_scores_map


# ==================== 主评估逻辑 ====================
def evaluate(cfg):
    print(f"[-] Loading model with vocab_size={cfg.vocab_size}...")
    model = LogBERT(cfg.vocab_size, cfg.hidden, cfg.layers, cfg.heads).to(cfg.device)

    # 加载权重
    try:
        checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)  # , weights_only=False)
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

    # 构造 Loader
    ds_norm = TailTestDataset(cfg.test_normal_file, tokenizer, cfg.max_len)
    ds_abnorm = TailTestDataset(cfg.test_abnormal_file, tokenizer, cfg.max_len)

    # 采样
    def get_subset(ds):
        total = len(ds)
        if cfg.test_ratio < 1.0:
            indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))[
                      :int(total * cfg.test_ratio)].tolist()
            return DataLoader(Subset(ds, indices), batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        return DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    loader_norm = get_subset(ds_norm)
    loader_abnorm = get_subset(ds_abnorm)

    # 候选参数
    k_candidates = [5, 10, 15, 20]

    print(f"[-] Extracting features (K={k_candidates})...")
    dist_n, topk_map_n = extract_features(model, center, loader_norm, cfg.device, tokenizer, k_candidates)
    dist_a, topk_map_a = extract_features(model, center, loader_abnorm, cfg.device, tokenizer, k_candidates)

    # 数据拼接 & 归一化
    y_true = np.concatenate([np.zeros(len(dist_n)), np.ones(len(dist_a))])

    # Z-Score Normalization
    raw_dist = np.concatenate([dist_n, dist_a])
    norm_dist = z_score_normalize(raw_dist)

    print("\n" + "=" * 60)
    print("GRID SEARCH: Z-Score(Dist) + Z-Score(TopK)")
    print("=" * 60)
    print(f"{'K':<3} | {'Alpha':<5} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'AUC':<7}")
    print("-" * 50)

    best_f1 = 0
    best_res = {}

    # 双重循环搜索
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for k in k_candidates:
        # 对 TopK 分数也做 Z-Score (因为不同 K 值下分数的均值方差也不同)
        raw_topk = np.concatenate([topk_map_n[k], topk_map_a[k]])
        norm_topk = z_score_normalize(raw_topk)

        for alpha in alphas:
            final_scores = alpha * norm_dist + (1 - alpha) * norm_topk

            p, r, t = precision_recall_curve(y_true, final_scores)
            f1s = 2 * p * r / (p + r + 1e-10)
            best_idx = np.argmax(f1s)

            curr_f1 = f1s[best_idx]

            # 只打印有潜力的结果 (F1 > 0.85)
            if curr_f1 > 0.85:
                curr_auc = roc_auc_score(y_true, final_scores)
                print(
                    f"{k:<3} | {alpha:<5.1f} | {p[best_idx]:.4f}  | {r[best_idx]:.4f}  | {curr_f1:.4f}  | {curr_auc:.4f}")

            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_res = {"k": k, "alpha": alpha, "f1": curr_f1, "auc": roc_auc_score(y_true, final_scores)}

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