# NLP 基于LSTM的文本分类

**本项目以cnews为数据，通过训练LSTM模型进行文本分类并对模型优化。**

![截圖 2024-01-19 18.34.46](https://cdn.jsdelivr.net/gh/Sean652039/pic_bed@main/uPic/%E6%88%AA%E5%9C%96%202024-01-19%2018.34.46.png)



## 文件结构

```bash
├───cnews
│   ├───train
│   ├───validation
│   └───test 
├───dataset
│   ├───train
│   ├───validation
│   └───test
├───logs
├───words
│   ├───words vector
│   └───stop words
├───scripts
```

项目的主要代码文件为 `train_news`。其中，`logs/` 包含训练过程的数据曲线（使用 Tensorboard 查看）；`cnews/` 是原始数据`dataset/` 是转换为token序列的数据，在重复训练时减少运行时间。



## 运行指南

- 本项目基于 Python 编程语言，用到的外部代码库主要包括jieba, PyTorch等。支持Apple Silicon加速。程序运行使用的 Python 版本为 3.10，建议使用 [Anaconda](https://www.anaconda.com/) 配置 Python 环境。以下配置过程已在 macOS Ventura 13.4 系统上测试通过。以下为控制台/终端（Console/Terminal/Shell）指令。

```bash
# 创建 conda 环境，将其命名为 NLP_Class，Python 版本 3.10
conda create -n NLP_Class python=3.10
conda activate NLP_Class

# 使用 GPU 训练需要手动安装 Apple Silicon 版 PyTorch
conda install pytorch::pytorch=2.0.1 torchvision torchaudio -c pytorch

# 降级安装外部代码库（针对macOS）
pip install setuptools==65.5.0 pip==21

pip install -r requirements.txt
```

- 下载相关文件

1. 下载cnews数据集并存放在`cnews`文件夹中。https://tianchi.aliyun.com/dataset/63227
2. 下载四川大学停止词放入`words`文件夹中。https://github.com/YueYongDev/stopwords
3. 下载腾讯AI实验室100维词向量放入`words`文件夹中。https://ai.tencent.com/ailab/nlp/en/download.html



## 训练模型

```bash
cd [项目上级文件夹]/NLP_classification
python train_news.py

Options:
  --epochs EPOCHS                     Number of training epochs (default: 20).
  --batch_size BATCH_SIZE             Batch size for training and evaluation (default: 16).
  --lr LR                             Learning rate for optimizer (default: 1e-3).
  --vocab_size VOCAB_SIZE             Vocabulary size (default: 6000).
  --padding_len PADDING_LEN           Padding length for sequences (default: 500).
  --device {cpu, cuda, mps}           Device for training and evaluation (choices: cpu, cuda, mps, default: mps).

Example:
  python train_news.py --epochs 20 --batch_size 16 --lr 1e-3 --vocab_size 6000 --padding_len 500 --device cuda
```



## 查看曲线

```bash
cd [项目上级文件夹]/NLP_Classification
tensorboard --logdir=logs/
```

在浏览器中打开 Tensorboard 服务默认地址 `http://localhost:6006/`，即可查看训练过程的曲线图以及模型。



## 鸣谢

- 本项目的基本框架和核心代码由华北电力大学王洪涛老师指导和提供，本人只对代码进一步完善和改进，包括对Apple Silicon的支持，训练过程的可视化以及更改了细微的错误，最后对代码进行一部分重构。
- 感谢四川大学的停止词以及腾讯AI实验室的词向量。
- 感谢B站up主林亦，本项目的重构其实是针对其AI贪吃蛇项目的学习（包括README）。
- 感谢各位开源工作者。