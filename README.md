# LogBERT: Log Anomaly Detection via BERT

### [ARXIV](https://arxiv.org/abs/2103.04475) 

This repository provides the implementation of Logbert for log anomaly detection. 
The process includes downloading raw data online, parsing logs into structured data, 
creating log sequences and finally modeling. 

![alt](img/log_preprocess.png)

## Configuration

- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.8
- PyTorch 1.9.0

## Installation

This code requires the packages listed in requirements.txt.
An virtual environment is recommended to run this code

On macOS and Linux:  

```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r ./environment/requirements.txt
deactivate
```

Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

An alternative is to create a conda environment:

```
    conda create -f ./environment/environment.yml
    conda activate logbert
```

Reference: https://docs.conda.io/en/latest/miniconda.html

## Experiment

Logbert and other baseline models are implemented on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [BGL](https://github.com/logpai/loghub/tree/master/BGL), and [thunderbird]() datasets

### HDFS example

```shell script
cd HDFS

sh init.sh

# process data
python data_process.py

#run logbert
python logbert.py vocab
python logbert.py train
python logbert.py predict

#run deeplog
python deeplog.py vocab
# set options["vocab_size"] = <vocab output> above
python deeplog.py train
python deeplog.py predict 

#run loganomaly
python loganomaly.py vocab
# set options["vocab_size"] = <vocab output> above
python loganomaly.py train
python loganomaly.py predict

#run baselines

baselines.ipynb
```

### Folders created during execution

```shell script 
~/.dataset //Stores original datasets after downloading
project/output //Stores intermediate files and final results during execution
```

### mac上传文件到server

```
ssh ...
# 上传文件
scp -P 端口号 本地文件路径 root@地址:/root/autodl-tmp/
# 上传文件夹
scp -P 端口号 -r 本地文件夹 root@地址:/root/autodl-tmp/
# 本地运行
scp -P 43355 /Users/wangzhi/Downloads/dataset/HDFS_v1.zip root@region-41.seetacloud.com:/root/.dataset/

```

### 监控AutoDL状态
```
# (实时动态查看GPU（每秒刷新一次）)
watch -n 1 nvidia-smi
# (实时动态查看CPU)
htop
```

### 代码结构
如果你的目标是 **SCI Q4+** 并且希望代码结构显得**专业、严谨**（符合 Hugging Face Transformers 的工程标准），那么随便把脚本扔在根目录是不行的。审稿人如果看到你的 GitHub 仓库结构混乱，第一印象分会大打折扣。

根据 **Hugging Face (HF)** 的标准项目结构，以及 Python 最佳实践，我为你设计了以下目录结构。

#### 1. 推荐的项目结构 (Project Structure)

Hugging Face 的项目通常遵循 **“配置(Config) - 数据(Data) - 模型(Model) - 脚本(Scripts)”分离** 的原则。

建议将你的项目命名为 `LogBERT-ParserFree`（或者更有学术感的名称），结构如下：

```text
BERT-pytorch/                  <-- 项目根目录
├── setup.py                   <-- [新增] 安装脚本，让 Python 认识 bert_pytorch 包
├── requirements.txt           <-- [新增] 依赖库列表
├── README.md                  <-- 项目说明
│
├── data/                      <-- 数据目录
│   ├── raw/                   <-- 存放原始日志 (HDFS.log, BGL.log)
│   └── processed/             <-- 存放处理后的 .txt 文件
│
├── scripts/                   <-- [新增] 运行脚本目录 (参照 HF examples)
│   └── run_preprocess.py      <-- 专门用于运行预处理的入口脚本
│
└── bert_pytorch/              <-- 核心源码包 (Library)
    ├── __init__.py            <-- 暴露包接口
    ├── dataset/
    │   ├── __init__.py
    │   ├── preprocessor.py    <-- 你的核心预处理类 (ParserFreePreprocessor)
    │   ├── dataset.py         <-- PyTorch Dataset 定义
    │   └── vocab.py           <-- 词表定义
    ├── model/
    │   ├── __init__.py
    │   ├── bert.py
    │   └── embedding/         <-- 存放 TimeEmbedding
    └── trainer/
        ├── __init__.py
        └── pretrain.py
```

#### 代码改进方案

1. Cursor+Composer 用OpenRouter代替解决方案
2. 超参数搜索
3. 保存每次训练、预测的超参数和F1

