import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import sys
import os
from hdsf.bm4_bert import LogBERT


# ================= 配置 =================
MODEL_PATH = "../output/hdfs/best_model.pth"
VOCAB_PATH = "../output/hdfs/"
TEST_NORMAL = "../output/hdfs/test_normal.csv"
TEST_ABNORMAL = "../output/hdfs/test_abnormal.csv"
MAX_LEN = 512  # 你的设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inspect_model():
    print(f"[-] Loading resources...")
    tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH, do_lower_case=True)
    vocab_size = len(tokenizer.vocab)

    model = LogBERT(vocab_size=vocab_size, hidden=128, n_layers=2).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)  # , weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')  # 不做 reduction，看每个词的 Loss

    def diagnose_samples(file_path, sample_type, num=3):
        print(f"\n{'=' * 20} Diagnosing {sample_type} Samples {'=' * 20}")
        df = pd.read_csv(file_path)
        data = df['EventSequence'].fillna("").tolist()[:num]  # 取前几条

        for i, text in enumerate(data):
            print(f"\n[Sample {i}] Original Length: {len(text.split())} words (approx)")

            # 1. 编码
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_LEN)
            input_ids = torch.tensor([tokens]).to(DEVICE)

            # 2. 检查截断情况
            token_list = tokenizer.convert_ids_to_tokens(tokens)
            print(f"-> Tokenized Length: {len(tokens)}")
            print(f"-> Last 5 tokens: {token_list[-5:]}")

            # 检查是否有异常关键词被保留
            keywords = ['failed', 'error', 'exception', 'terminate']
            found_keywords = [w for w in token_list if any(k in w for k in keywords)]
            print(f"-> Suspicious keywords found in input: {found_keywords}")

            # 3. 计算逐个 Token 的 Loss
            # 我们模拟预测过程：输入原图，看模型预测原图的概率
            with torch.no_grad():
                logits, _ = model(input_ids)  # [1, Seq, Vocab]

            # Shift label (像 GPT 一样，或者 BERT MLM)
            # 这里我们看 BERT 对自身的重构能力
            loss = criterion(logits.view(-1, vocab_size), input_ids.view(-1))
            loss = loss.view(1, -1)  # [1, Seq]

            # 4. 打印 Loss 最高的 Top-3 Token
            # 排除 [CLS], [SEP], [PAD]
            valid_mask = (input_ids != tokenizer.pad_token_id) & \
                         (input_ids != tokenizer.cls_token_id) & \
                         (input_ids != tokenizer.sep_token_id)

            masked_loss = loss * valid_mask.float()
            top_losses, top_indices = torch.topk(masked_loss, k=5)

            print(f"-> Max Loss: {top_losses[0][0].item():.4f}")
            print("-> Top-5 High Loss Tokens:")
            for j in range(5):
                idx = top_indices[0][j].item()
                val = top_losses[0][j].item()
                token_str = token_list[idx]
                print(f"   Rank {j + 1}: Token='{token_str}' (ID={tokens[idx]}) | Loss={val:.4f}")

    # 执行诊断
    diagnose_samples(TEST_NORMAL, "NORMAL", num=3)
    diagnose_samples(TEST_ABNORMAL, "ABNORMAL", num=3)


if __name__ == "__main__":
    inspect_model()