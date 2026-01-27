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


# ==================== 配置类 ====================
class Config:
    def __init__(self):
        # 路径配置 (请确保这些文件存在)
        self.output_dir = "../output/hdfs/"
        self.model_path = "../output/hdfs/best_model.pth"
        self.test_normal_file = "../output/hdfs/test_normal.csv"
        self.test_abnormal_file = "../output/hdfs/test_abnormal.csv"
        self.vocab_path = "../output/hdfs/"

        # TODO: 测试采样比例 (1.0 = 100% 全量测试)
        # 建议设为 1.0 以获得最终论文数据，调试时可设为 0.1
        self.test_ratio = 0.1

        # 模型参数 (必须与 train.py 一致)
        self.max_len = 128
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.0  # 推理时 Dropout 关闭

        # 推理性能配置
        self.batch_size = 512
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 测试专用 Dataset ====================
class TestDataset(Dataset):
    """
    轻量级测试 Dataset。
    只负责将文本转为 input_ids，不进行任何 Mask 或 Shuffle 操作。
    """

    def __init__(self, file_path, vocab_path, seq_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        self.seq_len = seq_len

        print(f"[-] Loading raw data from {file_path}...")
        self.df = pd.read_csv(file_path)

        # 兼容性处理
        if 'EventSequence' not in self.df.columns:
            # 尝试找第一列如果是文本
            print("Warning: 'EventSequence' column not found, using first column.")
            self.data = self.df.iloc[:, 0].fillna("").tolist()
        else:
            self.data = self.df['EventSequence'].fillna("").tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx])

        # 直接编码
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.seq_len)
        input_ids = tokens

        # Padding
        padding_len = self.seq_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long)
        }


# ==================== 核心算法：Max-Pooling Score ====================
def compute_scores_max_strategy(model, center, dataloader, device, tokenizer):
    """
    计算异常分数。
    核心改进：使用 Max Loss 而非 Mean Loss。
    """
    model.eval()
    # reduction='none' 保留每个 token 的 loss
    criterion_mlm = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    distance_scores = []
    mlm_max_scores = []  # 记录序列中的最大 Loss
    mlm_mean_scores = []  # 记录序列中的平均 Loss (用于对比)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference (Max-Strategy)"):
            raw_input = batch["input_ids"].to(device)

            # --- Metric 1: Distance (Global) ---
            _, pos_embedding = model(raw_input)
            # 计算欧氏距离
            dist = torch.norm(pos_embedding - center, dim=1)
            distance_scores.extend(dist.cpu().numpy().tolist())

            # --- Metric 2: MLM (Interleaved Full Coverage) ---
            # 存储每个 Token 的 Loss
            seq_token_losses = torch.zeros_like(raw_input, dtype=torch.float, device=device)

            # 标记有效的 token 位置 (非 PAD, CLS, SEP)
            valid_mask = (raw_input != pad_id) & (raw_input != cls_id) & (raw_input != sep_id)

            # 分两轮 Mask：Offset 0 (偶数位), Offset 1 (奇数位)
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
                labels[~token_mask] = -100  # 非 Mask 位置不计算 Loss

                # Forward
                mlm_logits, _ = model(masked_input)

                # Calc Loss
                vocab_size = mlm_logits.size(-1)
                loss_per_token = criterion_mlm(mlm_logits.view(-1, vocab_size), labels.view(-1))
                loss_per_token = loss_per_token.view(raw_input.size(0), -1)

                # 累加 Loss (因为两轮 Mask 的位置互斥，直接加即可)
                seq_token_losses += loss_per_token * token_mask.float()

            # --- Aggregation Strategies ---

            # 1. Max Strategy: 取一条日志中 Loss 最大的那个词的分数
            # 这种策略对 "Local Anomaly" (如 failed) 极其敏感
            batch_max_loss, _ = seq_token_losses.max(dim=1)
            mlm_max_scores.extend(batch_max_loss.cpu().numpy().tolist())

            # 2. Mean Strategy: 取平均 (仅除以有效 Token 数)
            valid_counts = valid_mask.sum(dim=1).clamp(min=1)
            batch_mean_loss = seq_token_losses.sum(dim=1) / valid_counts
            mlm_mean_scores.extend(batch_mean_loss.cpu().numpy().tolist())

    return distance_scores, mlm_max_scores, mlm_mean_scores


# ==================== 数据加载辅助 ====================
def get_subset_loader(dataset, cfg):
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
        print(f"[-] Sampling Subset: {cfg.test_ratio * 100}% ({subset_len}/{total_len})")
        indices = torch.randperm(total_len)[:subset_len].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(subset, **loader_kwargs)
    else:
        print(f"[-] Full Usage: {total_len} samples")
        return DataLoader(dataset, **loader_kwargs)


# ==================== 主评估流程 ====================
def evaluate(cfg):
    print(f"[-] Loading model from {cfg.model_path}")

    # 1. 准备 Dataset 和 Tokenizer
    # 使用 TestDataset 而不是 LogDataset
    ds_normal_raw = TestDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len)
    ds_abnormal_raw = TestDataset(cfg.test_abnormal_file, cfg.vocab_path, seq_len=cfg.max_len)

    tokenizer = ds_normal_raw.tokenizer
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

    # 安全加载
    checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    center = checkpoint['center'].to(cfg.device)
    print(f"[-] Model loaded. Center norm: {torch.norm(center).item():.4f}")

    # 3. 准备 Loader
    normal_loader = get_subset_loader(ds_normal_raw, cfg)
    abnormal_loader = get_subset_loader(ds_abnormal_raw, cfg)

    # 4. 计算分数
    print("[-] Computing scores...")
    n_dist, n_max, n_mean = compute_scores_max_strategy(model, center, normal_loader, cfg.device, tokenizer)
    a_dist, a_max, a_mean = compute_scores_max_strategy(model, center, abnormal_loader, cfg.device, tokenizer)

    y_true = [0] * len(n_dist) + [1] * len(a_dist)

    # ==================== 结果对比 ====================
    # 构造不同的评分策略字典
    strategies = {
        "Distance Only": np.array(n_dist + a_dist),
        "MLM (Mean)": np.array(n_mean + a_mean),
        "MLM (Max)": np.array(n_max + a_max),  # <--- 重点关注这个！
        "Hybrid (Max+Dist)": np.array(n_max + a_max) + 0.1 * np.array(n_dist + a_dist)
    }

    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION REPORT (Ratio={cfg.test_ratio})")
    print("=" * 60)

    results = []

    for name, y_scores in strategies.items():
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)

        # 找到最佳阈值
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_auc = roc_auc_score(y_true, y_scores)

        print(f"Strategy: {name:<20} | Best F1: {best_f1:.4f} | AUC: {best_auc:.4f}")

        results.append({
            "Strategy": name,
            "F1": best_f1,
            "AUC": best_auc
        })

    print("=" * 60)

    # 自动保存最佳结果到 CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(cfg.output_dir, "final_benchmark.csv"), index=False)
    print(f"[-] Benchmark saved to {cfg.output_dir}")


if __name__ == "__main__":
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)
    else:
        print(f"Model file not found at {cfg.model_path}")