import torch
import torch.nn as nn
from transformers import BertTokenizer
import os
import sys
import pandas as pd
from HDFS.bm4_bert import LogBERT


# 路径设置 (根据你的环境确认)
VOCAB_PATH = "../output/hdfs/"
MODEL_PATH = "../output/hdfs/best_model.pth"
NORMAL_FILE = "../output/hdfs/test_normal.csv"


def forensics():
    print(f"{'=' * 10} STEP 1: 环境一致性检查 {'=' * 10}")

    # 1. 检查 Vocab 文件
    vocab_file = os.path.join(VOCAB_PATH, "vocab.txt")
    if not os.path.exists(vocab_file):
        print(f"[!] Error: vocab.txt missing at {vocab_file}")
        return

    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_lines = f.readlines()
    vocab_size_file = len(vocab_lines)
    print(f"[-] vocab.txt 行数: {vocab_size_file}")

    # 检查 failed 是否存在
    failed_in_vocab = "failed\n" in vocab_lines
    print(f"[-] 'failed' 在词表中吗? {'YES' if failed_in_vocab else 'NO'}")

    # 2. 检查 Tokenizer 加载
    tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH, do_lower_case=True)
    print(f"[-] Tokenizer.vocab_size: {len(tokenizer.vocab)}")

    # 3. 检查模型权重
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Error: Model missing at {MODEL_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用 weights_only=False 以防万一
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # 获取模型里 Embedding 层的权重形状
    # state_dict key 可能是 'bert_mlm.bert.embeddings.word_embeddings.weight'
    # 我们遍历找一下
    embed_weight = None
    for k, v in checkpoint['model_state_dict'].items():
        if 'word_embeddings.weight' in k:
            embed_weight = v
            break

    if embed_weight is None:
        print("[!] 无法在 checkpoint 中找到 Embedding 权重，模型结构可能不对")
        return

    model_vocab_size = embed_weight.shape[0]
    print(f"[-] 模型内部 Embedding 大小: {model_vocab_size}")

    # 核心判断：匹配吗？
    if vocab_size_file != model_vocab_size:
        print(f"\n[!!! 致命错误 !!!] 词表文件有 {vocab_size_file} 个词，但模型是按 {model_vocab_size} 个词训练的！")
        print("原因: 你生成了新词表，但没有删除旧模型并重新训练。")
        print("后果: ID 错位。模型以为 ID 100 是 'Block'，但新词表里 ID 100 可能是 'IP'。")
        return
    else:
        print("[-] 维度匹配检查: PASS")

    print(f"\n{'=' * 10} STEP 2: 正常样本推理测试 {'=' * 10}")

    # 加载一条正常日志
    df = pd.read_csv(NORMAL_FILE)
    if 'EventSequence' not in df.columns:
        text = str(df.iloc[0, 0])
    else:
        text = str(df.iloc[0]['EventSequence'])

    print(f"测试文本 (Normal): {text[:100]}...")

    # 1. 编码
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)
    print(f"Token IDs: {tokens[:10]} ...")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)[:10]} ...")

    # 2. 模型预测
    model = LogBERT(vocab_size=model_vocab_size, hidden=128, n_layers=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_ids = torch.tensor([tokens]).to(device)

    # 简单算个 Loss (Max Strategy)
    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        logits, _ = model(input_ids)  # [1, Seq, Vocab]
        loss = criterion(logits.view(-1, model_vocab_size), input_ids.view(-1))

        # 排除 PAD
        valid = (input_ids != tokenizer.pad_token_id)
        loss = loss * valid.view(-1)

        print(f"\n[-] 逐词 Loss (Top 5):")
        top_loss, top_idx = torch.topk(loss, 5)
        for val, idx in zip(top_loss, top_idx):
            token_id = input_ids.view(-1)[idx].item()
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            print(f"    Token: '{token_str}' (ID {token_id}) -> Loss: {val.item():.4f}")

    avg_loss = loss.sum() / valid.sum()
    print(f"\n[-] 平均 Loss: {avg_loss.item():.4f}")

    if avg_loss > 1.0:
        print("\n[!!! 诊断结果 !!!] 正常样本 Loss 过高！")
        print("即使维度匹配，ID 映射也可能乱了。")
        print("建议：彻底删除 vocab.txt 和 best_model.pth，重新运行 train_tokenizer.py 和 train.py。")
    else:
        print("\n[OK] 正常样本 Loss 正常。如果 F1 还是低，可能是截断问题。")


if __name__ == "__main__":
    forensics()