import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset  # <--- [修改] 引入 Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score
import random

# 引入项目模块
from HDFS.bd3_dataset import LogDataset
from HDFS.bm4_bert import LogBERT


# ==================== 全局种子设置 (复现性保障) ====================
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
        # ... (之前的路径配置保持不变) ...
        self.output_dir = "../output/hdfs/"
        self.model_path = "../output/hdfs/best_model.pth"
        self.test_normal_file = "../output/hdfs/test_normal.csv"
        self.test_abnormal_file = "../output/hdfs/test_abnormal.csv"
        self.vocab_path = "../output/hdfs/"

        self.test_ratio = 0.1

        self.max_len = 128
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.0

        # [修改] 增大 Batch Size 和 Workers
        self.batch_size = 512  # 推理不占梯度显存，建议开大到 512 或 1024

        # Windows下建议设为 4，Linux下可以设为 8 或 16
        # 注意：num_workers > 0 时，首次加载会有几秒钟的“假死”用于启动子进程，是正常的
        self.num_workers = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 推理函数 ====================

def compute_scores(model, center, dataloader, device):
    """
    计算给定数据集中每个样本的异常分数。
    """
    model.eval()
    criterion_mlm = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    distances = []
    mlm_losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            # 1. 准备数据
            pos_input = batch["contrastive_pos"].to(device)
            bert_input = batch["bert_input"].to(device)
            bert_label = batch["bert_label"].to(device)

            # 2. Forward Pass
            mlm_logits, cls_embedding = model(bert_input)
            _, pos_embedding = model(pos_input)

            # 3. Metric 1: Distance to Center
            dist = torch.norm(pos_embedding - center, dim=1)
            distances.extend(dist.cpu().numpy().tolist())

            # 4. Metric 2: MLM Reconstruction Loss
            vocab_size = mlm_logits.size(-1)
            loss_per_token = criterion_mlm(mlm_logits.view(-1, vocab_size), bert_label.view(-1))
            loss_per_token = loss_per_token.view(pos_input.size(0), -1)
            loss_batch = loss_per_token.sum(dim=1) / (bert_label != -100).sum(dim=1).clamp(min=1)
            mlm_losses.extend(loss_batch.cpu().numpy().tolist())

    return distances, mlm_losses


def get_subset_loader(dataset, cfg, desc="Dataset"):
    """
    [修改] 支持 num_workers 和 pin_memory 的高性能 Loader
    """
    total_len = len(dataset)

    # 定义通用的 DataLoader 参数
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,  # 测试集不需要 shuffle
        "num_workers": cfg.num_workers,
        "pin_memory": True,  # [加速] 开启锁页内存，加速 CPU->GPU 传输
        "persistent_workers": True if cfg.num_workers > 0 else False  # [加速] 保持子进程活跃
    }

    if cfg.test_ratio < 1.0:
        subset_len = int(total_len * cfg.test_ratio)
        subset_len = max(1, subset_len)

        print(f"[-] {desc} Sampling: {cfg.test_ratio * 100}% ({subset_len}/{total_len})")

        # 之前修复的 tolist() bug 保留
        indices = torch.randperm(total_len)[:subset_len].tolist()
        subset = Subset(dataset, indices)

        return DataLoader(subset, **loader_kwargs)
    else:
        print(f"[-] {desc} Full Usage: {total_len} samples")
        return DataLoader(dataset, **loader_kwargs)


def evaluate(cfg):
    print(f"[-] Loading model from {cfg.model_path}")

    # 1. 临时加载 Tokenizer
    dummy_ds = LogDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len)
    vocab_size = len(dummy_ds.tokenizer.vocab)
    print(f"[-] Vocab Size: {vocab_size}")

    # 2. 加载模型
    model = LogBERT(
        vocab_size=vocab_size,
        hidden=cfg.hidden,
        n_layers=cfg.layers,
        attn_heads=cfg.heads,
        dropout=cfg.dropout
    ).to(cfg.device)

    checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    center = checkpoint['center'].to(cfg.device)
    print(f"[-] Model loaded. Center norm: {torch.norm(center).item():.4f}")

    # 3. 准备数据 Loader (使用 test_ratio 进行抽样)
    normal_ds = LogDataset(cfg.test_normal_file, cfg.vocab_path, seq_len=cfg.max_len)
    normal_loader = get_subset_loader(normal_ds, cfg, desc="Normal Test")

    abnormal_ds = LogDataset(cfg.test_abnormal_file, cfg.vocab_path, seq_len=cfg.max_len)
    abnormal_loader = get_subset_loader(abnormal_ds, cfg, desc="Abnormal Test")

    # 4. 计算分数
    print("[-] Computing scores for Normal samples...")
    norm_dists, norm_mlms = compute_scores(model, center, normal_loader, cfg.device)

    print("[-] Computing scores for Abnormal samples...")
    abnorm_dists, abnorm_mlms = compute_scores(model, center, abnormal_loader, cfg.device)

    # 5. 整合结果
    y_true = [0] * len(norm_dists) + [1] * len(abnorm_dists)

    # 主要汇报 Distance Score
    y_scores = norm_dists + abnorm_dists

    # 6. 计算指标
    print("[-] Calculating metrics...")
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx]
    best_prec = precision[best_idx]
    best_rec = recall[best_idx]

    auc_score = roc_auc_score(y_true, y_scores)

    print("\n" + "=" * 40)
    print(f"FINAL RESULTS (Ratio={cfg.test_ratio})")
    print("=" * 40)
    print(f"Best Threshold: {best_thresh:.4f}")
    print(f"Best F1-Score:  {best_f1:.4f}")
    print(f"Precision:      {best_prec:.4f}")
    print(f"Recall:         {best_rec:.4f}")
    print(f"AUC-ROC:        {auc_score:.4f}")
    print("=" * 40)

    # 保存结果时标记 ratio
    csv_name = f"test_results_ratio_{cfg.test_ratio}.csv"
    df_res = pd.DataFrame({
        "Label": y_true,
        "Distance_Score": y_scores,
        "MLM_Score": norm_mlms + abnorm_mlms
    })
    df_res.to_csv(os.path.join(cfg.output_dir, csv_name), index=False)
    print(f"[-] Detailed results saved to {csv_name}")


if __name__ == "__main__":
    # 固定种子
    set_seed(42)
    cfg = Config()
    if os.path.exists(cfg.model_path):
        evaluate(cfg)
    else:
        print("Model file not found!")