from numpy import dtype
from torch.utils.data import Dataset
import torch
import jieba
import jieba.posseg as pseg
import re
from tqdm import tqdm


def read_cnews_data(path):
    labels = []
    inputs = []
    label_num = {}
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if line[:2] not in label_num:
                label_num[line[:2]] = len(label_num)
            labels.append(label_num[line[:2]]) # 0, 1, 2, 3
            inputs.append(line[3:])
    print(label_num)
    return inputs, labels, len(label_num)


# 去除标点符号
# remove punctuation
def filter_punctuation(line):
    punc = "[！？。｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃" \
           "《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+"
    line = re.sub(punc, "", line)
    return line


class CnewsDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.avg_len = None
        self.inputs, self.labels, self.label_num = read_cnews_data(path)
        self.data2token()   # 现在的输入数据还是一整句话，首先要转为token序列，然后在训练的之前token要转为序号seq）
                            # Now the input data is still a whole sentence,
                            # which first has to be converted to a token sequence, and then the token
                            # has to be converted to an ordinal seq before training)

    def data2token(self):
        """
        原始文本转换为token
        """
        self.avg_len = 0
        for i, data in enumerate(tqdm(self.inputs)):
            # self.inputs[i] = jieba.lcut(filter_punctuation(data))
            seg = pseg.lcut(filter_punctuation(data), use_paddle=True)
            temp_str = []
            for seg, flag in seg:
                if flag not in ["nr", "ns", "nt", "nw", "t", "m"]:
                    temp_str.append(str(seg))
                else:
                    temp_str.append(str(flag))
            self.inputs[i] = temp_str
            self.avg_len += len(self.inputs[i])
        self.avg_len /= len(self.labels)
        print(f'the average length is {self.avg_len}')

    def token2seq(self, vocab, padding_len):
        """
        token转换为序号seq token to seq
        vocab：词典
        padding_len：输入句子的长度，截断或者填充
        """
        for i, data in enumerate(self.inputs):
            if len(self.inputs[i]) < padding_len:
                self.inputs[i] += [vocab.padding_word] * (padding_len - len(self.inputs[i]))
            elif(len(self.inputs[i]) > padding_len):
                self.inputs[i] = self.inputs[i][:padding_len]
            for j in range(padding_len):    # 所有token转序号
                self.inputs[i][j] = vocab.word2seq(self.inputs[i][j])
            self.inputs[i] = torch.tensor(self.inputs[i], dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item:int):
        return self.inputs[item], self.labels[item]


if __name__ == '__main__':
    from vocab import Vocab
    train_inputset = CnewsDataset(r'cnews/cnews.train.txt')
    vocab = Vocab(train_inputset.inputs, 1000)
    print(vocab.word2seq('天气'))