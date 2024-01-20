# NLP LSTM-based text classification

**In this project, we use cnews as data, and train LSTM model for text classification and optimize the model. **

![截圖 2024-01-19 18.34.46](https://cdn.jsdelivr.net/gh/Sean652039/pic_bed@main/uPic/%E6%88%AA%E5%9C%96%202024-01-19%2018.34.46.png)



## Structure

```bash
├───cnews
│   ├───train
│   ├───validation
│   ├───test
│   └───test witout labels
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

The main code file for the project is `train_news`. Where `logs/` contains the data curves of the training process (viewed using Tensorboard); `cnews/` is the raw data `dataset/` is the data converted into a sequence of tokens, to reduce the runtime when repeating the training.



## Configuration

- This project is based on Python programming language, using external code libraries including jieba, PyTorch and so on. It supports Apple Silicon acceleration. The program runs on Python version 3.10. It is recommended to use [Anaconda](https://www.anaconda.com/) to configure the Python environment. The following configuration procedure has been tested on a macOS Ventura 13.4 system. The following are Console/Terminal/Shell commands.

```bash
# Create a conda environment, name it NLP_Class, Python version 3.10
conda create -n NLP_Class python=3.10
conda activate NLP_Class

# Training with the GPU requires a manual installation of PyTorch for Apple Silicon.
conda install pytorch::pytorch=2.0.1 torchvision torchaudio -c pytorch

# Downgrading the installation of external code libraries(macOS only)
pip install setuptools==65.5.0 pip==21

pip install -r requirements.txt
```

- Download related files

1. Download the cnews dataset and put it in the `cnews` folder. https://tianchi.aliyun.com/dataset/63227
2. Download the Sichuan University stop words into the `words` folder. https://github.com/YueYongDev/stopwords
3. Download the Tencent AI Lab 100-dimensional word vector and put it in the `words` folder. https://ai.tencent.com/ailab/nlp/en/download.html



## Training

```bash
cd [Project parent folder]/NLP_classification
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



## Viewing Curve

```bash
cd [Project parent folder]/NLP_Classification
tensorboard --logdir=logs/
```

Open the Tensorboard service at the default address `http://localhost:6006/` in your browser to view the graphs of the training process as well as the model.



## Acknowledgements

- The basic framework and core code of this project were guided and provided by Dr. Hongtao Wang of North China Electric Power University, I only further refined and improved the code, including the support of Apple Silicon, the visualization of the training process as well as changed the minor errors, and finally partially refactored the code.
- Thanks to Sichuan University for stop words and Tencent AI Lab for word vectors.
- Thanks to Bilibili uploader Lin Yi, the refactoring of this project is actually for his AI Snake project learning (including this README).
- Thanks to all the open source workers.