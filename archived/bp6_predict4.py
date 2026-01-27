import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score
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


# ==================== 核心改进：Top-K 评分策略 ====================
def compute_scores_topk(model, dataloader, device, tokenizer, k=10):
    """
    计算 Top-K 异常分数。
    分数定义：序列中【不在 Top-K 预测范围内】的 Token 数量。
    如果一条日志里有 1 个词不在 Top-K，分数就是 1；如果有 3 个，分数就是 3。
    分数越高，越异常。
    """
    model.eval()

    # 结果存储
    anomaly_scores = []

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference (Top-{k})"):
            # 使用 contrastive_pos (原始序列)
            raw_input = batch["contrastive_pos"].to(device)
            batch_size, seq_len = raw_input.shape

            # 用于记录每条日志的累积异常计数
            # Shape: [Batch]
            batch_anomaly_counts = torch.zeros(batch_size, device=device)

            # 标记有效区域 (不评估 PAD, CLS, SEP)
            valid_mask = (raw_input != pad_id) & (raw_input != cls_id) & (raw_input != sep_id)

            # Interleaved Masking (两轮全覆盖)
            for offset in [0, 1]:
                masked_input = raw_input.clone()

                # 生成 Mask
                indices = torch.arange(seq_len, device=device)
                mask_cols = (indices % 2 == offset)
                token_mask = mask_cols.unsqueeze(0).expand_as(masked_input) & valid_mask

                masked_input[token_mask] = mask_id

                # Forward
                logits, _ = model(masked_input)  # [B, L, V]

                # 获取被 Mask 位置的 Top-K 预测
                # topk_indices: [B, L, K]
                _, topk_indices = torch.topk(logits, k=k, dim=-1)

                # 真实 Token: [B, L, 1]
                real_tokens = raw_input.unsqueeze(-1)

                # 检查真实值是否在 Top-K 中
                # hit: [B, L] (True=在, False=不在)
                hit = (topk_indices == real_tokens).any(dim=-1)

                # 统计异常：位置被Mask了 AND 没命中TopK
                # miss: [B, L] (1=异常, 0=正常)
                miss = token_mask & (~hit)

                # 累加异常词数量
                batch_anomaly_counts += miss.sum(dim=1).float()

            # 存入结果
            anomaly_scores.extend(batch_anomaly_counts.cpu().numpy().tolist())

    return np.array(anomaly_scores)


def get_subset_loader(dataset, cfg, desc="Dataset"):
    total_len = len(dataset)
    loader_kwargs = {"batch_size": cfg.batch_size, "shuffle": False, "num_workers": cfg.num_workers, "pin_memory": True}
    if cfg.test_ratio < 1.0:
        subset_len = int(total_len * cfg.test_ratio)
        subset_len = max(1, subset_len)
        indices = torch.randperm(total_len)[:subset_len].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(subset, **loader_kwargs)
    else:
        print(f"[-] {desc} Full Usage: {total_len} samples")
        return DataLoader(dataset, **loader_kwargs)


def evaluate(cfg):
    print(f"[-] Loading model from {cfg.model_path}")

    # 1. 准备 Dataset (获取 tokenizer)
    dummy_ds = LogDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len)
    tokenizer = dummy_ds.tokenizer
    vocab_size = len(tokenizer.vocab)
    print(f"[-] Vocab Size: {vocab_size}")

    # 2. 加载模型
    model = LogBERT(vocab_size=vocab_size, hidden=cfg.hidden, n_layers=cfg.layers, attn_heads=cfg.heads).to(cfg.device)
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    center = checkpoint['center'].to(cfg.device)  # 保留 center 虽然 TopK 不用它

    # 3. 准备数据
    norm_loader = get_subset_loader(LogDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len), cfg,
                                    "Normal")
    abnorm_loader = get_subset_loader(LogDataset(cfg.test_abnormal_file, cfg.vocab_path, seq_len=cfg.max_len), cfg,
                                      "Abnormal")

    # 4. 多 K 值扫描 (Grid Search for best K)
    # 我们尝试不同的 K 值，看哪个效果最好
    # HDFS 日志词汇少，K=3, 5, 10 应该足够
    k_candidates = [1, 3, 5, 10, 20]

    print("\n" + "=" * 60)
    print(f"TOP-K EVALUATION REPORT (Ratio={cfg.test_ratio})")
    print("=" * 60)

    for k in k_candidates:
        print(f"[-] Evaluating Top-{k} strategy...")
        n_scores = compute_scores_topk(model, norm_loader, cfg.device, tokenizer, k=k)
        a_scores = compute_scores_topk(model, abnorm_loader, cfg.device, tokenizer, k=k)

        y_true = [0] * len(n_scores) + [1] * len(a_scores)
        y_scores = np.concatenate([n_scores, a_scores])

        # 计算指标
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)

        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_thresh = thresholds[best_idx]  # 这里 thresh 是异常词的数量，通常是整数
        auc_score = roc_auc_score(y_true, y_scores)

        print(
            f"Strategy: Top-{k:<2} | Best F1: {best_f1:.4f} | AUC: {auc_score:.4f} | Thresh: >{int(best_thresh)} words")

    print("=" * 60)


if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)
    else:
        print("Model file not found!")