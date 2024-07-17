import numpy as np


class Accuracy:
    def __init__(self, vocab):
        self.vocab = {vocab[i].word: vocab[i] for i in range(len(vocab))}
        self.index2word = [None] * len(self.vocab)
        for i in range(len(self.vocab)):
            self.index2word[vocab[i].index] = vocab[i].word

        self.w0norm = None
        self.ok_vocab = None
        self.ok_index = None
        self.sections = None

    def normalize_vectors(self, v):
        return v / np.linalg.norm(v, ord=2, axis=-1).reshape((-1, 1))

    def init_sims(self):
        self.w0norm = self.normalize_vectors(self.model.w0)

    def most_similar(self, positive=[], negative=[], topn=10):
        positive = [(word, 1.0) if isinstance(word, str) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, str) else word for word in negative]
        all_words, mean = set(), []
        for word, weight in positive + negative:
            mean.append(weight * self.w0norm[self.vocab[word].index])
            # mean.append(weight * self.normalize_vectors(self.model.w0[self.vocab[word].index]).flatten())
            all_words.add(self.vocab[word].index)
        mean = self.normalize_vectors(np.asarray(mean).mean(axis=0)).astype(np.float32).flatten()
        dists = np.dot(self.w0norm, mean)
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        result = [(self.index2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]

    def prepare_data(self, questions, restrict_vocab=30000):
        ok_vocab = dict(sorted(self.vocab.items(), key=lambda item: -item[1].cn)[:restrict_vocab])
        ok_index = set(v.index for v in ok_vocab.values())

        sections = {}
        section = None
        words = []
        for line_no, line in enumerate(open(questions)):
            if line.startswith(': '):
                if section:
                    sections[section] = words
                    words = []
                section = line.lstrip(': ').strip()
            else:
                try:
                    a, b, c, expected = line.lower().split()
                except:
                    continue
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    continue
                words.append((a, b, c, expected))

        if section:
            sections[section] = words

        self.ok_vocab = ok_vocab
        self.ok_index = ok_index
        self.sections = sections

    def log_accuracy(self, section):
        correct, incorrect = section['correct'], section['incorrect']
        if correct + incorrect > 0:
            print("%s: %.1f%% (%i/%i)" % (
                section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect))

    def __call__(self, model):
        self.model = model
        self.init_sims()

        results = []
        for section in self.sections.keys():
            r = {'section': section, 'correct': 0, 'incorrect': 0}
            for a, b, c, expected in self.sections[section]:
                predicted, ignore = None, set(self.vocab[v].index for v in [a, b, c])
                for index in np.argsort(self.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
                    if index in self.ok_index and index not in ignore:
                        predicted = self.index2word[index]
                        if predicted != expected:
                            pass
                        break
                r['correct' if predicted == expected else 'incorrect'] += 1
            results.append(r)

        total = {
            'section': 'total',
            'correct': np.sum(s['correct'] for s in results),
            'incorrect': np.sum(s['incorrect'] for s in results)
        }
        self.log_accuracy(total)
        results.append(total)

        return results