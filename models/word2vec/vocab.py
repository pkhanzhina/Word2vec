import numpy as np
from tqdm import tqdm


class vocab_word:
    def __init__(self, word):
        self.word = word
        self.cn = 0
        self.point = None
        self.code = None
        self.codelen = None
        self.index = None


class Vocab:

    def __init__(self, cfg):
        self.vocab = []
        self.vocab_size = 0
        self.word2index = {}

        self.min_reduce = 1
        self.min_count = cfg.min_count

        self.MAX_CODE_LENGTH = cfg.MAX_CODE_LENGTH
        self.vocab_max_size = cfg.vocab_max_size

    def SearchVocab(self, word):
        return self.word2index.get(word, -1)

    def AddWordToVocab(self, word):
        self.vocab.append(vocab_word(word))
        self.vocab_size += 1
        self.word2index[word] = self.vocab_size - 1
        return self.vocab_size - 1

    def ReduceVocab(self):
        self.vocab = [self.vocab[a] for a in range(self.vocab_size) if self.vocab[a].cn > self.min_reduce]
        self.vocab_size = len(self.vocab)
        self.word2index = {self.vocab[a].word: a for a in range(self.vocab_size)}
        self.min_reduce += 1

    def SortVocab(self):
        self.vocab = sorted(self.vocab, key=lambda x: x.cn, reverse=True)
        self.vocab = [self.vocab[a] for a in range(self.vocab_size) if self.vocab[a].cn >= self.min_count]
        self.vocab_size = len(self.vocab)
        self.word2index = {self.vocab[a].word: a for a in range(self.vocab_size)}

        for a in range(self.vocab_size):
            self.vocab[a].code = np.zeros(self.MAX_CODE_LENGTH).astype(np.uint8)
            self.vocab[a].point = np.zeros(self.MAX_CODE_LENGTH).astype(np.uint32)
            self.vocab[a].index = a

    def LearnVocabFromTrainFile(self, file):
        train_words = 0
        self.vocab_size = 0
        for w in tqdm(file, desc='Prepare vocab'):
            if len(w) == 0:
                continue
            train_words += 1
            i = self.SearchVocab(w)
            if i == -1:
                a = self.AddWordToVocab(w)
                self.vocab[a].cn = 1
            else:
                self.vocab[i].cn += 1
            if self.vocab_size > self.vocab_max_size * 0.7:
                self.ReduceVocab()
        self.SortVocab()
        print('Vocab size:', self.vocab_size)

    def CreateBinaryTree(self):
        point       = np.zeros(self.MAX_CODE_LENGTH,    dtype=np.uint32)
        code        = np.zeros(self.MAX_CODE_LENGTH,    dtype=np.uint8)
        count       = np.zeros(self.vocab_size * 2 + 1, dtype=np.int64)
        binary      = np.zeros(self.vocab_size * 2 + 1, dtype=np.int64)
        parent_node = np.zeros(self.vocab_size * 2 + 1, dtype=np.int64)

        for a in range(self.vocab_size):
            count[a] = self.vocab[a].cn
        for a in range(self.vocab_size, self.vocab_size * 2):
            count[a] = 1e15
        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size
        for a in range(self.vocab_size - 1):
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1
            count[self.vocab_size + a] = count[min1i] + count[min2i]
            parent_node[min1i] = self.vocab_size + a
            parent_node[min2i] = self.vocab_size + a
            binary[min2i] = 1
        for a in range(self.vocab_size):
            b, i = a, 0
            while True:
                code[i] = binary[b]
                point[i] = b
                i += 1
                b = parent_node[b]
                if b == self.vocab_size * 2 - 2:
                    break
            self.vocab[a].codelen = i
            self.vocab[a].point[0] = self.vocab_size - 2
            for b in range(i):
                self.vocab[a].code[i - b - 1] = code[b]
                self.vocab[a].point[i - b] = point[b] - self.vocab_size
        del count, binary, parent_node

    def GetSentence(self, sentence):
        res = []
        for w in tqdm(sentence, desc='Get sentence'):
            if len(w) == 0:
                res.append(None)
                continue
            i = self.SearchVocab(w)
            if i == -1:
                res.append(None)
            else:
                res.append(self.vocab[i])
        return res


if __name__ == '__main__':
    pass