class Vocab:
    def __init__(self, datas: list, limit_size) -> None:
        self.stop_words = None
        self.vocab = {}
        self.padding_word = '<pad>'
        self.set_stopword()
        cnt = {}
        for data in datas:
            for word in data:
                if word not in cnt: cnt[word] = 1
                else: cnt[word] += 1
        self.vocab[self.padding_word] = 0
        if len(cnt) > limit_size:
            cnt = sorted([_ for _ in cnt.items() if _[0] not in self.stop_words], key=lambda t: t[1],
                     reverse=True)
            for w, _ in cnt:
                if len(self.vocab) == limit_size: break
                self.vocab[w] = len(self.vocab)
        else:
            pass

    def __len__(self):
        return len(self.vocab)

    def word2seq(self, word) -> int:
        if word not in self.vocab:
            return self.vocab[self.padding_word]
        return self.vocab[word]

    def set_stopword(self, path='words/scu_stopwords.txt'):
        """
        设置停止词
        set the stop words
        """
        with open(path, 'r', encoding='utf-8') as fr:
            self.stop_words = [line.strip() for line in fr.readline()]
