import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增引用
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from hdfs.bd3_dataset import LogDataset
from hdfs.bm4_bert import LogBERT


def set_seed(seed=42):
    """固定随机种子，确保论文实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[-] Random seed set to {seed}")


class Config:
    def __init__(self):
        # 路径配置
        self.output_dir = "../output/hdfs/"
        self.train_file = "../output/hdfs/train.csv"
        # vocab.txt 所在目录
        self.vocab_path = "../output/hdfs/"

        # TinyBERT 超参 (论文核心配置)
        self.max_len = 512
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.1

        # 训练配置
        self.batch_size = 64
        self.epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-4

        # TODO: Trinity Loss 权重 [修改点 2] 大幅降低 SVDD 权重
        # 让模型优先学好 MLM (语义)，SVDD 只是辅助
        self.w_svdd = 0.05     # 从 0.5 改为 0.05(防止坍塌)
        self.w_simcse = 0.05  # 从 0.1 改为 0.05
        self.temp_simcse = 0.05

        # 验证与早停
        self.patience = 3
        self.val_ratio = 0.1  # 10% 用于验证

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)


def init_center(model, dataloader, device):
    """使用正常样本初始化超球体中心"""
    print("[-] Initializing Hypersphere Center...")
    model.eval()
    center = torch.zeros(model.config.hidden_size, device=device)
    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init Center"):
            input_ids = batch["contrastive_pos"].to(device)
            _, embeddings = model(input_ids)
            center += torch.sum(embeddings, dim=0)
            n_samples += input_ids.size(0)
    center /= n_samples

    # 防止中心为0
    if torch.norm(center) < 1e-5:
        center += 1e-4

    print(f"[-] Center initialized. Norm: {torch.norm(center).item():.4f}")
    return center


def save_model(model, center, epoch, path):
    """保存模型状态和球心向量"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'center': center.cpu()
    }, path)


# [新增] SimCSE Loss 计算函数
def compute_simcse_loss(z1, z2, temperature=0.05):
    batch_size = z1.size(0)
    sim_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2) / temperature
    labels = torch.arange(batch_size).to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def validate(model, val_loader, center, criterion_mlm, device, cfg):
    """计算验证集 Loss (MLM + SVDD)"""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            bert_input = batch["bert_input"].to(device)
            bert_label = batch["bert_label"].to(device)
            pos_input = batch["contrastive_pos"].to(device)
            # neg_input 在 SVDD 模式下不需要

            mlm_logits, _ = model(bert_input)
            _, pos_emb = model(pos_input)

            vocab_size = mlm_logits.size(-1)
            # 1. MLM Loss
            mlm_loss = criterion_mlm(mlm_logits.view(-1, vocab_size), bert_label.view(-1))

            # 2. SVDD Loss (MSE)
            dist_sq = torch.sum((pos_emb - center) ** 2, dim=1)
            svdd_loss = torch.mean(dist_sq)

            # 验证集不加 SimCSE，只关注聚类效果
            loss = mlm_loss + cfg.w_svdd * svdd_loss

            batch_size = bert_input.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train(cfg):
    print(f"[-] Training on {cfg.device}")

    # 1. 加载数据
    full_train_dataset = LogDataset(cfg.train_file, cfg.vocab_path, seq_len=cfg.max_len)
    total_samples = len(full_train_dataset)

    # 划分验证集
    val_size = int(total_samples * cfg.val_ratio)
    train_size = total_samples - val_size
    print(f"[-] Splitting: Train={train_size}, Val={val_size}")

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # drop_last=True 对 SimCSE 很重要
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)

    # 2. 模型初始化
    vocab_size = len(full_train_dataset.tokenizer.vocab)
    print(f"[-] Vocab Size: {vocab_size}")

    model = LogBERT(
        vocab_size=vocab_size,
        hidden=cfg.hidden,
        n_layers=cfg.layers,
        attn_heads=cfg.heads,
        dropout=cfg.dropout
    ).to(cfg.device)

    # 3. 初始化 Center
    center = init_center(model, train_loader, cfg.device)

    # 4. 优化器与 Loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    # [修改点] 移除了 TripletLoss

    best_val_loss = float('inf')
    patience_counter = 0

    # 5. 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0
        total_batches = 0

        # 监控 MLM 准确率
        total_correct = 0
        total_masked = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Train]")

        for batch in progress_bar:
            bert_input = batch["bert_input"].to(cfg.device)
            bert_label = batch["bert_label"].to(cfg.device)
            pos_input = batch["contrastive_pos"].to(cfg.device)
            # neg_input = batch["contrastive_neg"].to(cfg.device) # 不再使用

            # Forward
            # 1. MLM 任务
            mlm_logits, _ = model(bert_input)

            # 2. SimCSE + SVDD 任务 (两次 Forward 获取 Dropout 差异)
            _, emb1 = model(pos_input)
            _, emb2 = model(pos_input)

            # Loss Calculation
            # A. MLM
            mlm_loss = criterion_mlm(mlm_logits.view(-1, vocab_size), bert_label.view(-1))

            # B. SVDD (让 emb1 靠近 center)
            dist_sq = torch.sum((emb1 - center) ** 2, dim=1)
            svdd_loss = torch.mean(dist_sq)

            # C. SimCSE (让 emb1 和 emb2 相似，且与其他样本不相似)
            simcse_loss = compute_simcse_loss(emb1, emb2, cfg.temp_simcse)

            # Total Loss
            loss = mlm_loss + (cfg.w_svdd * svdd_loss) + (cfg.w_simcse * simcse_loss)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # [新增] Center 软更新 (防止 Center 死板)
            with torch.no_grad():
                batch_center = torch.mean(emb1, dim=0)
                center = 0.995 * center + 0.005 * batch_center

            total_train_loss += loss.item()
            total_batches += 1

            # 计算准确率 (仅针对 Masked Token)
            with torch.no_grad():
                preds = mlm_logits.argmax(dim=-1)
                mask = (bert_label != -100)
                if mask.sum() > 0:
                    correct = (preds == bert_label) & mask
                    total_correct += correct.sum().item()
                    total_masked += mask.sum().item()

            acc = total_correct / total_masked if total_masked > 0 else 0

            # 保持原有的打印信息
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MLM_Acc": f"{acc:.2%}"
            })

        avg_train_loss = total_train_loss / total_batches

        # Validation
        avg_val_loss = validate(model, valid_loader, center, criterion_mlm, cfg.device, cfg)

        print(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train Acc={total_correct / total_masked:.2%}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            print(f"    [*] Saving new best model...")
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(cfg.output_dir, "best_model.pth")
            save_model(model, center, epoch, save_path)
        else:
            patience_counter += 1
            print(f"    [!] Patience: {patience_counter}/{cfg.patience}")
            if patience_counter >= cfg.patience:
                print("    [STOP] Early stopping triggered.")
                break


if __name__ == "__main__":
    set_seed(42)
    config = Config()
    if not os.path.exists(config.train_file):
        print(f"Error: Train file not found: {config.train_file}")
        exit(1)
    train(config)