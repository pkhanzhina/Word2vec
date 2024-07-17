from easydict import EasyDict
import os

model_cfg = EasyDict()
model_cfg.hidden_size = 200
model_cfg.window_size = 7

model_cfg.vocab = EasyDict()
model_cfg.vocab.min_count = 5
model_cfg.vocab.MAX_CODE_LENGTH = 40
model_cfg.vocab.vocab_max_size = 30000000

ROOT_DIR = ''
dataset_cfg = EasyDict()
dataset_cfg.data_path = os.path.join(ROOT_DIR, 'data\Text8\text8')
dataset_cfg.qa_path = os.path.join(ROOT_DIR, 'data\word2vec\questions-words.txt')

train_cfg = EasyDict()
train_cfg.init_lr = 0.025
train_cfg.last_lr = 0.0001
train_cfg.sentence_len = 1000
train_cfg.eval_freq = 2500

train_cfg.with_sigmoid_table = True
train_cfg.with_blas = True

train_cfg.dataset = dataset_cfg
train_cfg.model = model_cfg
