import torch
from vocab import Vocab
from torch import nn
import numpy as np


def read_pretrained_wordvec(path, vocab:Vocab, word_dim):
    """
    给vocab中的每个词分配词向量,如果有预先传入的训练好的词向量,则提取出来 Assign word vectors to each word in the vocab, if there are pre-passed trained word vectors, they are extracted
    path: 词向量存储路径
    vocab: 词典
    word_dim: 词向量的维度
    返回值是词典（按照序号）对应的词向量 The return value is the word vector corresponding to the dictionary (by ordinal number)
    """
    vecs = np.random.normal(0.0, 0.9, [len(vocab), word_dim]) # 先随机给词典中的每个词分一个随机词向量
    with open(path, 'r') as file:
        for line in file:
            line = line.split()
            if line[0] in vocab.vocab:  # 在词典里则提取出来，存到序号对应的那一行去
                vecs[vocab.word2seq(line[0])] = np.asarray(line[1:], dtype='float32')
    return vecs


class MyLSTM(nn.Module):
    def __init__(self, vecs, vocab_size, word_dim, num_layer, hidden_size, label_num) -> None:
        super(MyLSTM, self).__init__()
        # (1) 随机生成词向量，然后训练过程中再通过梯度更新调整词向量
        # (1) Randomly generate word vectors, and then adjust the word vectors by gradient updating during the training process
        # self.embedding_layer = nn.Embedding(vocab_size, word_dim)
        # self.embedding_layer.weight.requires_grad = True

        # (2)读取预训练的词向量，训练过程不更新梯度
        # (2) Read pre-trained word vectors, training process does not update gradient
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(vecs).float()) # 原来没加.float()导致mac上后面出现类型错误
        self.embedding_layer.weight.requires_grad = False
        
        self.rnn = nn.LSTM(word_dim, hidden_size, num_layer)
        """
        随机丢弃，dropout是指在深度学习网络的训练过程中，对于神经网络单元，
        按照一定的概率将其暂时从网络中丢弃。某些隐含层节点的权重不工作,不工作的那些节点可以暂时认为不是网络结构的一部分,
        但是它的权重得保留下来。防止过拟合提高效果。
        Random dropout, dropout means that during the training process of a deep learning network, 
        for neural network units, they are temporarily dropped from the network according to a certain probability. 
        The weights of certain hidden layer nodes do not work, and those that do not work can be 
        temporarily considered not to be part of the network structure, but its weights have to be retained. 
        Preventing overfitting improves results
        """
        self.fc = nn.Sequential(
            nn.Dropout(0.5),    #
            nn.Linear(hidden_size, label_num)
        )

    def forward(self, X):
        # [batch, seq, word_vec] -> [seq, batch, word_vec]
        # permute 可以将tensor的维度换位，参数表示原来的维度下标
        # permute permutes the dimension of the tensor, with the parameter indicating the original dimension subscripts
        X = X.permute(1, 0)

        # 建立词向量层
        # Build word vector layers
        X = self.embedding_layer(X)

        # 先喂给LSTM
        # Pass it to LSTM first
        outs, _ = self.rnn(X)

        # LSTM的输出中的最后一个cell的输出喂给全连接层做预测
        # The output of the last cell in the output of the LSTM is fed to the fully connected layer to make predictions
        logits = self.fc(outs[-1])
        return logits
