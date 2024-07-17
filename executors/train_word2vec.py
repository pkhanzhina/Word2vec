import re
import sys
import time

import numpy as np
import pyximport
from tqdm import tqdm

from models.word2vec.model import Model
from utils.similar_accuracy import Accuracy

pyximport.install(setup_args={'include_dirs': np.get_include()})
from c_functions.word2vec_blas_sigmoid_cython import train_sentence as train_with_blas_sigmoid
from c_functions.word2vec_sigmoid_cython import train_sentence as train_with_sigmoid
from c_functions.word2vec_blas_cython import train_sentence as train_with_blas
from c_functions.word2vec_cython import train_sentence as train_cython


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.__prepare_data()
        self.__prepare_model()

        if self.cfg.with_sigmoid_table and self.cfg.with_blas:
            self.train_func = train_with_blas_sigmoid
            print('train with sigmoid and blas')
        elif self.cfg.with_sigmoid_table and not self.cfg.with_blas:
            self.train_func = train_with_sigmoid
            print('train with sigmoid')
        elif not self.cfg.with_sigmoid_table and self.cfg.with_blas:
            self.train_func = train_with_blas
            print('train with blas')
        else:
            self.train_func = train_cython
            print('train with pure sython')

    def __prepare_data(self):
        with open(self.cfg.dataset.data_path, 'r') as f:
            data = f.read()

        data = data.lower()
        self.train_data = re.sub(r"[^a-z0-9 ]", "", data).split(' ')

    def __prepare_model(self):
        self.model = Model(self.cfg.model)
        self.model.init_vocab(self.train_data)

        self.compute_accuracy = Accuracy(self.model.vocab.vocab)
        self.compute_accuracy.prepare_data(self.cfg.dataset.qa_path)

    def train_epoch(self):
        sentences = self.model.vocab.GetSentence(self.train_data)
        nrof_batches = len(sentences) // self.cfg.sentence_len + (len(sentences) % self.cfg.sentence_len != 0)
        lrs = np.linspace(self.cfg.last_lr, self.cfg.init_lr, nrof_batches)[::-1]

        test_accuracy = {0: self.evaluate()}
        for _iter in tqdm(range(nrof_batches)):
            s = _iter * self.cfg.sentence_len
            f = min((_iter + 1) * self.cfg.sentence_len, len(sentences))

            self.train_func(self.model, sentences[s:f], lrs[_iter])
            if _iter > 0 and _iter % self.cfg.eval_freq == 0:
                test_accuracy[_iter] = self.evaluate()
        test_accuracy[nrof_batches] = self.evaluate()
        return test_accuracy

    def evaluate(self):
        acc = self.compute_accuracy(self.model)
        return acc

    def one_batch_training(self, train_func, nrof_experiments):
        sentences = self.model.vocab.GetSentence(self.train_data)
        t = []
        for i in range(nrof_experiments):
            s = time.time()
            train_func(self.model, sentences[i * 1000: (i + 1) * 1000], self.cfg.last_lr)
            f = time.time()
            t.append(f - s)
        return np.round(1000 / np.mean(t))


if __name__ == '__main__':
    from configs.word2vec_cfg import train_cfg
    from utils.visualization import plot_accuracy

    trainer = Trainer(train_cfg)
    accuracy = trainer.train_epoch()
    plot_accuracy(accuracy, path_to_save='data/word2vec/accuracy_plot_by_sections.png')
    plot_accuracy(accuracy, ['total'], path_to_save='data/word2vec/accuracy_plot_total.png')

    # timing = {}
    # nrof_experiments = 10
    # for func_name in ['train_with_blas_sigmoid', 'train_with_sigmoid', 'train_with_blas', 'train_cython']:
    #     trainer = Trainer(train_cfg)
    #     timing[func_name] = trainer.one_batch_training(getattr(sys.modules[__name__], func_name), nrof_experiments)
    #
    # for key in timing.keys():
    #     print(key, timing[key])
