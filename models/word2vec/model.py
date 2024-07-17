import numpy as np
from models.word2vec.vocab import Vocab


class Model:
    def __init__(self, cfg):

        self.cfg = cfg

        self.hidden_size = cfg.hidden_size
        self.window = cfg.window_size

        self.vocab, self.w0, self.w1 = None, None, None

    def init_vocab(self, data):
        self.vocab = Vocab(self.cfg.vocab)
        self.vocab.LearnVocabFromTrainFile(data)
        self.vocab.CreateBinaryTree()

        self.init_layers()

    def init_layers(self):
        self.w0 = np.random.uniform(-0.5, 0.5, (self.vocab.vocab_size, self.hidden_size)) / self.hidden_size
        self.w1 = np.zeros((self.hidden_size, self.vocab.vocab_size))
        self.w0 = self.w0.astype(np.float32)
        self.w1 = self.w1.astype(np.float32)