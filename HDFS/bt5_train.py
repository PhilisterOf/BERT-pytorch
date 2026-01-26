import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from HDFS.bd3_dataset import LogDataset
from HDFS.bm4_bert import LogBERT


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
        self.max_len = 128
        self.hidden = 128
        self.layers = 2
        self.heads = 4
        self.dropout = 0.1

        # 训练配置
        self.batch_size = 64
        self.epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-4

        # Loss 权重
        self.contrastive_weight = 1.0
        self.margin = 0.5

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
    # 归一化可以增加训练稳定性，但在 SimCSE 中保持模长通常更好
    print(f"[-] Center initialized. Norm: {torch.norm(center).item():.4f}")
    return center


def save_model(model, center, epoch, path):
    """保存模型状态和球心向量"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'center': center.cpu()  # 移至 CPU 保存，通用性更好
    }, path)


def validate(model, val_loader, center, criterion_mlm, criterion_triplet, device, cfg):
    """计算验证集 Loss"""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            bert_input = batch["bert_input"].to(device)
            bert_label = batch["bert_label"].to(device)
            pos_input = batch["contrastive_pos"].to(device)
            neg_input = batch["contrastive_neg"].to(device)

            mlm_logits, _ = model(bert_input)
            _, pos_emb = model(pos_input)
            _, neg_emb = model(neg_input)

            vocab_size = mlm_logits.size(-1)
            # 计算验证 Loss
            mlm_loss = criterion_mlm(mlm_logits.view(-1, vocab_size), bert_label.view(-1))

            batch_center = center.unsqueeze(0).expand(pos_emb.size(0), -1)
            cont_loss = criterion_triplet(batch_center, pos_emb, neg_emb)

            loss = mlm_loss + cfg.contrastive_weight * cont_loss

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
    criterion_triplet = nn.TripletMarginLoss(margin=cfg.margin, p=2)

    best_val_loss = float('inf')
    patience_counter = 0

    # 5. 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0
        total_batches = 0

        # [新增] 监控 MLM 准确率
        total_correct = 0
        total_masked = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Train]")

        for batch in progress_bar:
            bert_input = batch["bert_input"].to(cfg.device)
            bert_label = batch["bert_label"].to(cfg.device)
            pos_input = batch["contrastive_pos"].to(cfg.device)
            neg_input = batch["contrastive_neg"].to(cfg.device)

            # Forward
            mlm_logits, _ = model(bert_input)
            _, pos_emb = model(pos_input)
            _, neg_emb = model(neg_input)

            # Loss Calculation
            mlm_loss = criterion_mlm(mlm_logits.view(-1, vocab_size), bert_label.view(-1))

            batch_center = center.unsqueeze(0).expand(pos_emb.size(0), -1)
            cont_loss = criterion_triplet(batch_center, pos_emb, neg_emb)

            loss = mlm_loss + cfg.contrastive_weight * cont_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_batches += 1

            # [新增] 计算准确率 (仅针对 Masked Token)
            with torch.no_grad():
                preds = mlm_logits.argmax(dim=-1)
                mask = (bert_label != -100)
                if mask.sum() > 0:
                    correct = (preds == bert_label) & mask
                    total_correct += correct.sum().item()
                    total_masked += mask.sum().item()

            acc = total_correct / total_masked if total_masked > 0 else 0

            # 实时显示 Acc，方便判断模型是否在学习
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MLM_Acc": f"{acc:.2%}"
            })

        avg_train_loss = total_train_loss / total_batches

        # Validation
        avg_val_loss = validate(model, valid_loader, center, criterion_mlm, criterion_triplet, cfg.device, cfg)

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