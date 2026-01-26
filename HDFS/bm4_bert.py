import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM


class LogBERT(nn.Module):
    """
    改进版 LogBERT 模型 (Lightweight + Semantic-Aware Pooling)

    论文创新点支撑:
    1. Efficiency: 采用 TinyBERT 架构，大幅降低参数量，适配边缘侧日志分析。
    2. Representation: 引入 Attention Pooling 替代静态的 [CLS] 标记，
       实现对日志中关键语义词（Semantic Triggers）的动态聚焦，抑制变量噪声。
    """

    def __init__(self, vocab_size, hidden=128, n_layers=2, attn_heads=4, dropout=0.1):
        """
        Args:
            vocab_size: 词表大小 (来自 vocab.txt, 约 3000)
            hidden: 隐藏层维度 (默认 128, 原版 BERT 是 768)
            n_layers: Transformer 层数 (默认 2, 原版 BERT 是 12)
            attn_heads: 注意力头数
        """
        super().__init__()

        # 1. 定义 TinyBERT 配置 (From Scratch)
        # 不加载预训练权重，完全针对日志数据从头训练
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden,
            num_hidden_layers=n_layers,
            num_attention_heads=attn_heads,
            intermediate_size=hidden * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=512,
            type_vocab_size=2
        )

        # 2. 初始化 Backbone
        # 使用 HuggingFace 官方实现的 Transformer 结构，保证计算稳定性
        self.bert_mlm = BertForMaskedLM(self.config)

        # TODO: 3. [创新点] Semantic-Aware Attention Pooling
        # 相比 SimCSE 直接取 [CLS] 或 Mean Pooling，
        # Attention Pooling 能让模型自动学会忽略 [BLK], [NUM] 等无意义变量
        self.attention_weights = nn.Linear(hidden, 1)

        # 4. 最终映射层 (Projection Head)
        # 将加权聚合后的向量映射到对比学习空间
        self.final_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),  # 非线性激活，增加表达能力
            nn.Linear(hidden, hidden)
        )

    def forward(self, input_ids, segment_label=None):
        """
        Args:
            input_ids: (Batch, Seq_Len)
        Returns:
            mlm_logits: (Batch, Seq_Len, Vocab) -> 用于 MLM Loss (重建局部语法)
            cls_embedding: (Batch, Hidden)      -> 用于 Contrastive Loss (学习全局时序)
        """

        # [Fix: 消除警告的核心修改]
        # 自动构建 Attention Mask: 非 0 位置为 1 (真实Token)，0 位置为 0 (Padding)
        # 这告诉 BERT 的 Attention 机制不要去关注补零的地方
        attention_mask = (input_ids != 0).long()

        # 1. Backbone Forward
        # output_hidden_states=True 让我们能拿到最后一层的序列输出，而不是仅仅是 logits
        outputs = self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,  # <--- 将 mask 传入这里
            output_hidden_states=True
        )

        mlm_logits = outputs.logits  # (Batch, Seq, Vocab)

        # 获取最后一层的隐状态序列: (Batch, Seq, Hidden)
        last_hidden = outputs.hidden_states[-1]

        # TODO: 改进点: --- 步骤 A: 计算原始分数 (Scoring) ---
        # self.attention_weights 是一个 Linear(hidden, 1) 层
        # 它给序列中的每个 Token 打分，判断它有多重要
        # last_hidden: [Batch, Seq, Hidden] -> scores: [Batch, Seq, 1]
        # scores.shape: [B, L, H] * [H, 1] = [B, L, 1]
        scores = self.attention_weights(last_hidden)

        # --- 步骤 B: 排除干扰 (Masking) ---
        # 对于 Padding (补零) 的位置，强制赋予负无穷大的分数
        # 这样 Softmax 之后它们的权重就是 0，不会影响结果
        # input_ids.shape: [B, L]
        # mask.unsqueeze(-1): [B, L, 1]
        mask = (input_ids == 0).unsqueeze(-1)
        scores = scores.masked_fill(mask, -1e9)

        # --- 步骤 C: 归一化权重 (Softmax) ---
        # 将分数转化为概率分布 (所有 Token 权重之和为 1)
        #   dim 0 (Batch): 第几条日志。
        #   我们不希望跨日志归一化（比如不能因为第一条日志很重要，就压低第二条日志的权重）。每条日志是独立的。
        #   dim 1 (Sequence): 日志里的第几个词 ([CLS], failed, block...)。
        #   这才是我们要比较的对象！ 我们想知道：“在这句话里，failed 和 block 谁更重要？”
        #   所以我们要在这个维度上让概率之和等于 1。
        #   dim 2 (Feature): 分数维度（大小为 1）。
        #   这里只有一个数值，在这个维度做 Softmax 毫无意义（结果永远是 1.0）。
        #   比如: "failed": 0.8, "block": 0.1, "[NUM]": 0.05, ...
        attn_probs = F.softmax(scores, dim=1)

        # --- 步骤 D: 加权聚合 (Weighted Sum) ---
        # 这就是 "Pooling" (池化) 的本质
        # 不是简单取平均 (Mean Pooling)，也不是只取第一个 (CLS Pooling)
        # 而是根据重要性加权求和

        # 1. last_hidden: [B, L, H]  (例如: 2, 10, 128)
        #    这是 10 个词，每个词都是 128维向量。

        # 2. attn_probs:  [B, L, 1]  (例如: 2, 10, 1)
        #    这是 10 个词的权重，比如 [0.1, 0.8, 0.1 ...]

        # 3. 乘法 (Broadcasting): last_hidden * attn_probs
        #    [2, 10, 128] * [2, 10, 1] -> [2, 10, 128]
        #    物理意义：把“failed”这个词的向量乘以 0.8 (放大)，把“block”的向量乘以 0.1 (缩小)。
        #    形状依然是 [2, 10, 128]。

        # 4. 求和: torch.sum(..., dim=1)
        #    dim=1 是 Sequence 维度 (也就是那个 10)。
        #    我们沿着这个维度把 10 个加权后的向量加起来，变成 1 个向量。
        # [B, L, H] -> [B, H]
        sentence_embedding = torch.sum(last_hidden * attn_probs, dim=1)

        # 3. 映射到对比空间
        cls_embedding = self.final_projection(sentence_embedding)

        return mlm_logits, cls_embedding


# ==================== 单元测试 ====================
if __name__ == "__main__":
    # 简单的冒烟测试，确保维度变换正确
    vocab_size = 3000
    model = LogBERT(vocab_size=vocab_size, hidden=128, n_layers=2)

    # 模拟输入: Batch=2, Seq=10
    # 假设最后两个位置是 0 (Padding)
    input_ids = torch.randint(1, vocab_size, (2, 10))
    input_ids[:, -2:] = 0

    logits, embedding = model(input_ids)

    print("[-] Model Forward Check:")
    print(f"    Input Shape: {input_ids.shape}")
    print(f"    MLM Logits:  {logits.shape} (Expect: Batch, Seq, Vocab)")
    print(f"    Embedding:   {embedding.shape} (Expect: Batch, Hidden)")

    if logits.shape == (2, 10, 3000) and embedding.shape == (2, 128):
        print("[-] Test Passed!")
    else:
        print("[!] Test Failed.")