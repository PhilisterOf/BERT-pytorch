import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 导入我们自定义的模块
# 假设你的目录结构是标准的 src.data 和 src.models
from HDFS.bd3_dataset import LogDataset
from HDFS.bm4_bert import LogBERT


# ==================== 配置类 ====================
class Config:
    def __init__(self):
        # 路径配置
        self.output_dir = "../output/hdfs/"
        self.train_file = "../output/hdfs/train.csv"
        self.test_file = "../output/hdfs/test_normal.csv"  # 仅用于验证 MLM Loss
        self.vocab_path = "../output/hdfs/"  # vocab.txt 所在目录

        # 训练超参 (TinyBERT 设置)
        self.max_len = 128
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.1

        # 优化器配置
        self.batch_size = 64  # 显存够可调大
        self.epochs = 20  # HDFS 较小，20轮足够
        self.lr = 1e-3  # AdamW 常用 LR
        self.weight_decay = 1e-4

        # Loss 权重
        # total_loss = mlm_loss + lambda * contrastive_loss
        self.contrastive_weight = 1.0
        self.margin = 0.5  # Triplet Loss 的边界

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)


# ==================== 核心工具函数 ====================

def init_center(model, dataloader, device):
    """
    [Trick] 在训练前初始化超球体中心。
    计算所有正常训练样本 Embedding 的均值，作为初始 Center。
    这比随机初始化 Center 收敛快得多。
    """
    print("[-] Initializing Hypersphere Center...")
    model.eval()
    center = torch.zeros(model.config.hidden_size, device=device)
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init Center"):
            # 我们使用 contrastive_pos (未 Mask 的正常序列) 来计算中心
            input_ids = batch["contrastive_pos"].to(device)
            _, embeddings = model(input_ids)

            center += torch.sum(embeddings, dim=0)
            n_samples += input_ids.size(0)

    center /= n_samples

    # 归一化 Center (可选，视具体距离度量而定，这里不做强制归一化以保留模长信息)
    print(f"[-] Center initialized. Norm: {torch.norm(center).item():.4f}")
    return center


def save_model(model, center, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'center': center  # 必须保存 Center，因为它是异常检测的基准点
    }, path)


# ==================== 训练循环 ====================

def train(cfg):
    print(f"[-] Training on {cfg.device}")

    # 1. 准备数据
    train_dataset = LogDataset(cfg.train_file, cfg.vocab_path, seq_len=cfg.max_len)
    # drop_last=True 防止最后一个 batch 大小不一致导致 Center 计算抖动
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # 2. 初始化模型
    # 自动读取 vocab.txt 获取大小
    vocab_size = len(train_dataset.tokenizer.vocab)
    print(f"[-] Vocab Size: {vocab_size}")

    model = LogBERT(
        vocab_size=vocab_size,
        hidden=cfg.hidden,
        n_layers=cfg.layers,
        attn_heads=cfg.heads,
        dropout=cfg.dropout
    ).to(cfg.device)

    # 3. 初始化 Center (关键步骤)
    center = init_center(model, train_loader, cfg.device)

    # 4. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Loss 定义
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)  # 自动忽略 Mask

    # Triplet Margin Loss: max(0, dist_pos - dist_neg + margin)
    # 目标：让 Pos 离 Center 近，Neg 离 Center 远
    criterion_triplet = nn.TripletMarginLoss(margin=cfg.margin, p=2)

    # 5. 开始 Epochs
    best_loss = float('inf')

    for epoch in range(cfg.epochs):
        model.train()
        total_mlm_loss = 0
        total_cont_loss = 0
        total_correct = 0
        total_tokens = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for batch in progress_bar:
            # 数据搬运
            bert_input = batch["bert_input"].to(cfg.device)  # [B, L] (Masked)
            bert_label = batch["bert_label"].to(cfg.device)  # [B, L]
            pos_input = batch["contrastive_pos"].to(cfg.device)  # [B, L] (Original)
            neg_input = batch["contrastive_neg"].to(cfg.device)  # [B, L] (Shuffled)

            # --- Forward Pass 1: MLM ---
            # 主要用于学习 Context
            mlm_logits, _ = model(bert_input)

            # (B, L, V) -> (B*L, V) 用于 CrossEntropy
            mlm_loss = criterion_mlm(mlm_logits.view(-1, vocab_size), bert_label.view(-1))

            # --- Forward Pass 2: Contrastive ---
            # 主要用于学习 Global Representation
            # 共享参数，但是输入不同
            _, pos_emb = model(pos_input)  # [B, H]
            _, neg_emb = model(neg_input)  # [B, H]

            # 构造 Triplet Loss
            # Anchor: Center (所有正常样本的归宿)
            # Positive: 正常的日志 (应该靠近 Center)
            # Negative: 乱序的日志 (应该远离 Center)
            # 注意: Center 需要扩展 batch 维度
            batch_center = center.unsqueeze(0).expand(pos_emb.size(0), -1)  # [B, H]

            # TripletLoss(anchor, positive, negative)
            # Loss = max(0, d(center, pos) - d(center, neg) + margin)
            cont_loss = criterion_triplet(batch_center, pos_emb, neg_emb)

            # --- Backward ---
            loss = mlm_loss + cfg.contrastive_weight * cont_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Metrics & Logging ---
            total_mlm_loss += mlm_loss.item()
            total_cont_loss += cont_loss.item()

            # 计算 MLM 准确率 (仅计算被 Mask 的位置)
            mask = (bert_label != -100)
            if mask.sum() > 0:
                predictions = mlm_logits.argmax(dim=-1)
                correct = (predictions == bert_label) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

            acc = total_correct / total_tokens if total_tokens > 0 else 0

            progress_bar.set_postfix({
                "MLM": f"{mlm_loss.item():.4f}",
                "Cont": f"{cont_loss.item():.4f}",
                "Acc": f"{acc:.2%}"
            })

        # End of Epoch
        avg_loss = (total_mlm_loss + total_cont_loss) / len(train_loader)
        print(f"Epoch {epoch + 1} Summary: Avg Loss={avg_loss:.4f}, MLM Acc={total_correct / total_tokens:.2%}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(cfg.output_dir, "best_model.pth")
            save_model(model, center, epoch, save_path)
            print(f"[*] Best model saved to {save_path}")

        # 每一个 Epoch 结束后，动态更新一点点 Center?
        # LogBERT 原文建议: c = c + 0.1 * (mean_pos_emb - c)
        # 这是一个 Deep SVDD 的 trick，防止 Center 也是随机初始化的导致难以收敛
        # 但我们之前做过 init_center 了，所以这里不需要大幅更新，保持固定通常更稳。


if __name__ == "__main__":
    # 使用配置类管理参数
    config = Config()

    # 检查数据是否存在
    if not os.path.exists(config.train_file):
        print(f"Error: Train file not found: {config.train_file}")
        print("Please run data preprocessing scripts first.")
        exit(1)

    train(config)