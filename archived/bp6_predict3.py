import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import random
from HDFS.bd3_dataset import LogDataset
from HDFS.bm4_bert import LogBERT



# ==================== 全局种子设置 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[-] Random seed set to {seed}")


# ==================== 配置 ====================
class Config:
    def __init__(self):
        self.output_dir = "../output/hdfs/"
        self.model_path = "../output/hdfs/best_model.pth"
        self.test_normal_file = "../output/hdfs/test_normal.csv"
        self.test_abnormal_file = "../output/hdfs/test_abnormal.csv"
        self.vocab_path = "../output/hdfs/"

        # 建议使用 1.0 (全量) 以获得论文最终数据
        self.test_ratio = 0.1

        self.max_len = 128
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.0

        # 推理性能配置
        self.batch_size = 512
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 核心改进：Max-Strategy 推理 ====================
def compute_scores_max_strategy(model, center, dataloader, device, tokenizer):
    """
    [Key Fix]
    1. 使用 'contrastive_pos' (原始未Mask序列) 替代 'bert_input' (随机序列)。
    2. 使用 Interleaved Masking (全覆盖) 替代 Random Masking。
    3. 使用 Max Loss 替代 Mean Loss。
    """
    model.eval()
    criterion_mlm = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    distance_scores = []
    mlm_max_scores = []  # 存 Max
    mlm_mean_scores = []  # 存 Mean (对比用)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference (Max-Strat)"):
            # [Fix 1] 使用原始数据，不要用 Dataset 里随机 mask 过的 bert_input
            raw_input = batch["contrastive_pos"].to(device)

            # --- Metric 1: Distance (Global) ---
            _, pos_embedding = model(raw_input)
            dist = torch.norm(pos_embedding - center, dim=1)
            distance_scores.extend(dist.cpu().numpy().tolist())

            # --- Metric 2: MLM (Interleaved Full Coverage) ---
            # 存储每个 Token 的 Loss
            seq_token_losses = torch.zeros_like(raw_input, dtype=torch.float, device=device)

            # 标记有效区域 (非 Pad, Cls, Sep)
            valid_mask = (raw_input != pad_id) & (raw_input != cls_id) & (raw_input != sep_id)

            # [Fix 2] 分两轮 Mask：Offset 0 (偶数位), Offset 1 (奇数位)
            # 确保每个词都被预测过一次，没有任何随机性
            for offset in [0, 1]:
                masked_input = raw_input.clone()
                labels = raw_input.clone()

                # 生成 Mask
                seq_len = raw_input.size(1)
                indices = torch.arange(seq_len, device=device)
                mask_cols = (indices % 2 == offset)

                # 应用 Mask: 必须是偶数位 AND 有效位
                token_mask = mask_cols.unsqueeze(0).expand_as(masked_input) & valid_mask

                masked_input[token_mask] = mask_id
                labels[~token_mask] = -100

                # Forward
                mlm_logits, _ = model(masked_input)

                # Calc Loss
                vocab_size = mlm_logits.size(-1)
                loss_per_token = criterion_mlm(mlm_logits.view(-1, vocab_size), labels.view(-1))
                loss_per_token = loss_per_token.view(raw_input.size(0), -1)

                # 累加 Loss
                seq_token_losses += loss_per_token * token_mask.float()

            # [Fix 3] Max Strategy
            # 取一条日志中 Loss 最大的那个词作为该日志的异常分
            batch_max_loss, _ = seq_token_losses.max(dim=1)
            mlm_max_scores.extend(batch_max_loss.cpu().numpy().tolist())

            # Mean Strategy (仅作对比)
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)
            batch_mean_loss = seq_token_losses.sum(dim=1) / valid_counts
            mlm_mean_scores.extend(batch_mean_loss.cpu().numpy().tolist())

    return distance_scores, mlm_max_scores, mlm_mean_scores


# ==================== Loader ====================
def get_subset_loader(dataset, cfg, desc="Dataset"):
    total_len = len(dataset)
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": True
    }

    if cfg.test_ratio < 1.0:
        subset_len = int(total_len * cfg.test_ratio)
        subset_len = max(1, subset_len)
        print(f"[-] {desc} Sampling: {cfg.test_ratio * 100}% ({subset_len}/{total_len})")
        indices = torch.randperm(total_len)[:subset_len].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(subset, **loader_kwargs)
    else:
        print(f"[-] {desc} Full Usage: {total_len} samples")
        return DataLoader(dataset, **loader_kwargs)


# ==================== Main ====================
def evaluate(cfg):
    print(f"[-] Loading model from {cfg.model_path}")

    # 1. 临时加载 Tokenizer (用于获取 mask_id 等)
    dummy_ds = LogDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len)
    tokenizer = dummy_ds.tokenizer
    vocab_size = len(tokenizer.vocab)
    print(f"[-] Vocab Size: {vocab_size}")

    # 2. 加载模型
    model = LogBERT(
        vocab_size=vocab_size,
        hidden=cfg.hidden,
        n_layers=cfg.layers,
        attn_heads=cfg.heads,
        dropout=cfg.dropout
    ).to(cfg.device)

    # weights_only=True 消除安全警告
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)  # , weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    center = checkpoint['center'].to(cfg.device)
    print(f"[-] Model loaded. Center norm: {torch.norm(center).item():.4f}")

    # 3. 准备数据
    # 注意：这里我们加载 LogDataset，但只用它的 'contrastive_pos' 字段
    normal_loader = get_subset_loader(LogDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len), cfg,
                                      "Normal")
    abnormal_loader = get_subset_loader(LogDataset(cfg.test_abnormal_file, cfg.vocab_path, seq_len=cfg.max_len), cfg,
                                        "Abnormal")

    # 4. 计算分数
    print("[-] Computing scores...")
    n_dist, n_max, n_mean = compute_scores_max_strategy(model, center, normal_loader, cfg.device, tokenizer)
    a_dist, a_max, a_mean = compute_scores_max_strategy(model, center, abnormal_loader, cfg.device, tokenizer)

    y_true = [0] * len(n_dist) + [1] * len(a_dist)

    # ==================== 结果对比 ====================
    strategies = {
        "Distance Only": np.array(n_dist + a_dist),
        "MLM (Mean)": np.array(n_mean + a_mean),
        "MLM (Max)": np.array(n_max + a_max),  # <--- 期望冠军
        "Hybrid (Max+Dist)": np.array(n_max + a_max) + 0.1 * np.array(n_dist + a_dist)
    }

    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION REPORT (Ratio={cfg.test_ratio})")
    print("=" * 60)

    results = []
    for name, y_scores in strategies.items():
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)

        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_auc = roc_auc_score(y_true, y_scores)

        print(f"Strategy: {name:<20} | Best F1: {best_f1:.4f} | AUC: {best_auc:.4f}")
        results.append({"Strategy": name, "F1": best_f1, "AUC": best_auc})

    print("=" * 60)

    # 保存结果
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(cfg.output_dir, "final_benchmark.csv"), index=False)
    print(f"[-] Benchmark saved to {cfg.output_dir}")


if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)
    else:
        print("Model file not found!")