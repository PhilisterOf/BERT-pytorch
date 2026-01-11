# src/lad_bert/data/preprocessor.py
import re
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm


class ParserFreePreprocessor:
    def __init__(self, dataset_type='hdfs', window_size=20, step_size=5):
        self.dataset_type = dataset_type.lower()
        self.window_size = window_size
        self.step_size = step_size

    def clean_log_content(self, content):
        # ... (保持之前的清洗逻辑不变) ...
        content = content.lower()
        content = re.sub(r'(\d{1,3}\.){3}\d{1,3}', '[IP]', content)
        if 'blk_' in content:
            content = re.sub(r'blk_[-0-9]+', '[BLK]', content)
        content = re.sub(r'\b\d+\b', '[NUM]', content)
        content = re.sub(r'([^\w\s\[\]])', r' \1 ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        return content

    def process_hdfs(self, log_file):
        print(f"正在处理 HDFS (带时间): {log_file} ...")

        # 1. 定义正则：同时提取日期、时间、BlockID、Content
        # 样本: 081109 203518 143 INFO ...
        # Group 1: Date (081109)
        # Group 2: Time (203518)
        # Group 3: Block ID
        # Group 4: Content (剩下的)
        pattern = re.compile(r'^(\d{6})\s+(\d{6}).*?(blk_[-0-9]+)(.*)')

        grouped_data = {}  # {blk_id: [{'timestamp': float, 'content': str}, ...]}

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f):
                match = pattern.search(line)
                if match:
                    date_str, time_str, blk_id, raw_content = match.groups()

                    # 2. 解析时间戳
                    # HDFS 格式: yymmdd HHMMSS
                    dt = datetime.strptime(f"{date_str}{time_str}", "%y%m%d%H%M%S")
                    timestamp = dt.timestamp()

                    # 3. 清洗文本
                    cleaned_content = self.clean_log_content(raw_content)

                    if blk_id not in grouped_data:
                        grouped_data[blk_id] = []

                    grouped_data[blk_id].append({
                        'timestamp': timestamp,
                        'content': cleaned_content
                    })

        print("分组完成，开始计算时间间隔...")

        final_sequences_text = []
        final_sequences_time = []

        # 4. 对每个 Block 内部排序并计算 Delta T
        for blk_id, logs in grouped_data.items():
            # 按时间排序
            logs.sort(key=lambda x: x['timestamp'])

            # 提取文本列表
            text_seq = [x['content'] for x in logs]

            # 计算时间间隔 (当前时间 - 上一条日志时间)
            # 第一条日志间隔为 0
            time_seq = [0.0]
            for i in range(1, len(logs)):
                delta = logs[i]['timestamp'] - logs[i - 1]['timestamp']
                # 为了防止时间过大影响模型，通常取 log 或者截断，这里先保留原始值
                time_seq.append(float(f"{delta:.4f}"))

            final_sequences_text.append(text_seq)
            final_sequences_time.append(time_seq)

        return final_sequences_text, final_sequences_time

    def process_bgl(self, log_file):
        """
        BGL 专用处理 (也适用于 Thunderbird，格式非常像)
        BGL 格式: Label Timestamp Date Node Time NodeRepeated Type Content...
        例子: - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ...
        """
        print(f"正在处理 {self.dataset_type.upper()} (滑动窗口 + 时间): {log_file} ...")

        all_logs = []  # 存储所有解析好的 {'timestamp': t, 'content': c}

        # BGL 的时间戳通常在第二列 (index 1)，是 Unix Timestamp
        # 如果你的 BGL 数据格式不同，这里可能要微调

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="读取原始日志"):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # 1. 提取 Label (可选，如果是训练集，通常只保留 Label 为 '-' 的正常日志)
                label = parts[0]
                # if label != '-': continue # 如果只想训练正常日志，把这行注释打开

                # 2. 提取时间戳 (BGL 的第2列通常是 Unix 时间戳)
                try:
                    timestamp = float(parts[1])
                except ValueError:
                    continue  # 格式错误的行跳过

                # 3. 提取 Content
                # BGL 的正文通常从第 9 列或者第 10 列开始，前面都是元数据
                # 为了偷懒且保证信息全，我们可以取第 5 列往后的所有内容
                raw_content = " ".join(parts[4:])

                # 4. 清洗
                cleaned_content = self.clean_log_content(raw_content)

                all_logs.append({
                    'timestamp': timestamp,
                    'content': cleaned_content
                })

        # 按照时间排序 (防止原始日志乱序)
        all_logs.sort(key=lambda x: x['timestamp'])

        print(f"共读取 {len(all_logs)} 条日志，开始执行滑动窗口切分...")
        print(f"窗口大小: {self.window_size}, 步长: {self.step_size}")

        final_texts = []
        final_times = []

        # === 核心算法：滑动窗口生成 Time Embedding ===
        num_logs = len(all_logs)
        for i in tqdm(range(0, num_logs - self.window_size, self.step_size), desc="生成窗口"):
            # 切片
            window = all_logs[i: i + self.window_size]

            # 1. 文本序列
            text_seq = [x['content'] for x in window]

            # 2. 时间序列 (计算 Delta T)
            # 窗口内的第一条日志，Delta T 设为 0
            time_seq = [0.0]
            for j in range(1, len(window)):
                delta = window[j]['timestamp'] - window[j - 1]['timestamp']
                # 同样的，保留浮点数
                time_seq.append(float(f"{delta:.4f}"))

            final_texts.append(text_seq)
            final_times.append(time_seq)

        return final_texts, final_times

    def process(self, input_path):
        if self.dataset_type == 'hdfs':
            return self.process_hdfs(input_path)
        elif self.dataset_type in ['bgl', 'thunderbird']:
            return self.process_bgl(input_path)
        else:
            raise ValueError(f"未知数据集: {self.dataset_type}")

    def process(self, input_path):
        if self.dataset_type == 'hdfs':
            return self.process_hdfs(input_path)
        else:
            # BGL/TB 的时间处理逻辑类似，需解析它们特定的时间格式
            raise NotImplementedError("BGL/TB 时间解析暂未添加，请先跑 HDFS")

# 贴在 preprocessor.py 最下面
if __name__ == "__main__":
    # === 1. 设置路径 ===
    # 这里的 ../.. 取决于你运行脚本时所在的目录。
    # 如果你在 src/lad_bert/data/ 下运行，这样写是对的。
    TEST_INPUT = "../../data/raw/HDFS.log"
    TEST_OUTPUT_DIR = "../../data/processed"

    if not os.path.exists(TEST_INPUT):
        print(f"❌ 测试失败：找不到文件 {TEST_INPUT}")
        # 建议打印一下当前工作目录，方便排查路径问题
        print(f"当前工作目录: {os.getcwd()}")
    else:
        # 创建输出文件夹
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

        # === 2. 运行处理 ===
        p = ParserFreePreprocessor(dataset_type='hdfs')
        texts, times = p.process(TEST_INPUT)

        # 预览
        print(f"✅ 处理完成，内存中生成了 {len(texts)} 条数据")

        # === 3. 保存文件 (这是你之前缺失的部分！) ===
        # 保存文本文件
        out_text_path = os.path.join(TEST_OUTPUT_DIR, "hdfs_corpus_text.txt")
        print(f"正在保存文本到: {out_text_path}")
        with open(out_text_path, 'w', encoding='utf-8') as f:
            for seq in texts:
                # 用 [SEP] 拼接
                f.write(" [SEP] ".join(seq) + "\n")

        # 保存时间文件
        out_time_path = os.path.join(TEST_OUTPUT_DIR, "hdfs_corpus_time.txt")
        print(f"正在保存时间到: {out_time_path}")
        with open(out_time_path, 'w', encoding='utf-8') as f:
            for seq in times:
                # 用逗号拼接数字
                f.write(",".join(map(str, seq)) + "\n")

        print("🎉 全部保存完毕！")

# # === 测试代码 ===
# if __name__ == "__main__":
#     # 假设你下载了 BGL 的前 2000 行做测试
#     TEST_INPUT = "../../data/raw/BGL_2k.log"
#     TEST_OUTPUT_DIR = "../../data/processed"
#
#     # 为了测试 BGL，记得实例化时指明 dataset_type
#     p = ParserFreePreprocessor(dataset_type='bgl', window_size=20, step_size=5)
#
#     # ... (后面的保存代码和之前一样) ...