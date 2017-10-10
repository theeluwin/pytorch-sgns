# -*- coding: utf-8 -*-

import codecs
import pickle


class Preprocess(object):

    def __init__(self, window=2, unk='UNK'):
        self.window = window
        self.unk = unk
        self.vocab = set()
        self.word2idx = {}
        self.ready = False

    def tokenize(self, line):
        tokens = [token.strip() for token in line.split() if token.strip()]
        if self.ready:
            unked_tokens = []
            for token in tokens:
                if token in self.vocab:
                    unked_tokens.append(token)
                else:
                    unked_tokens.append(self.unk)
            tokens = unked_tokens
        return tokens

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self, filepath):
        print("building vocab...")
        step = 0
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                print("working on line {}".format(step), end='\r')
                sentence = self.tokenize(line)
                for word in sentence:
                    self.vocab.add(word)
        for idx, word in enumerate(self.vocab):
            self.word2idx[word] = idx + 1
        self.word2idx[self.unk] = 0
        self.ready = True
        pickle.dump(self.vocab, open('./data/vocab.dat', 'wb'))
        pickle.dump(self.word2idx, open('./data/word2idx.dat', 'wb'))
        print("\nbuild done")

    def convert(self, filepath):
        print("converting corpus...")
        step = 0
        data = []
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                print("working on line {}".format(step), end='\r')
                sentence = self.tokenize(line)
                for i in range(len(sentence)):
                    iword, owords = self.skipgram(sentence, i)
                    data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
        pickle.dump(data, open('./data/train.dat', 'wb'))
        print("\nconversion done")


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.build('./data/corpus.txt')
    preprocess.convert('./data/corpus.txt')
