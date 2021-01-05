from transformer import Constants
import itertools
from collections import Counter


class Vocabulary():

    def __init__(self, min_freq=1, max_size=5000):
        self.pad_token = Constants.PAD_WORD
        self.pad_idx = 0
        self.bos_token = Constants.BOS_WORD
        self.bos_idx = 1
        self.eos_token = Constants.EOS_WORD
        self.eos_idx = 2
        self.unk_token = Constants.UNK_WORD
        self.unk_idx = 3
        self.stoi = {self.pad_token: self.pad_idx, self.bos_token: self.bos_idx,
                     self.eos_token: self.eos_idx, self.unk_token: self.unk_idx}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.min_freq = min_freq
        self.max_size = max_size
        self.vocab_size = None

    '''
    input: data is list sentence which sentence is list of word
    '''
    def build_vocab(self, sentences, lower=True):
        words = list(itertools.chain.from_iterable(sentences))
        if lower:
            words = list(map(lambda w: w.lower(), words))
        word_freq = Counter(words)
        word_freq = word_freq.most_common(self.max_size)
        words = [w[0] for w in word_freq if w[1] > self.min_freq]
        for w in words:
            idx = len(self.stoi)
            self.stoi[w] = idx
            self.itos[idx] = w
        self.vocab_size = len(self.stoi)
