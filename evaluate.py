# -*- coding: utf-8 -*-

import csv
import pickle
import codecs
import numpy as np

from numpy.linalg import norm


word2idx = pickle.load(open('./data/word2idx.dat', 'rb'))
idx2vec = pickle.load(open('./data/idx2vec.dat', 'rb'))


def w2v(w):
    return idx2vec[word2idx[w]]


def cos(v1, v2):
    return v1.dot(v2) / norm(v1) / norm(v2)


def evaluate():
    golds = []
    with codecs.open('./data/gold.txt', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            golds.append({
                'v1': row[0],
                'v2': row[1],
                'sim': float(row[2]),
            })
    x = np.array([gold['sim'] for gold in golds])
    y = np.array([cos(w2v(gold['v1']), w2v(gold['v2'])) for gold in golds])
    p = cos(x, y)
    print("pearson correlation: {:.2f}%".format(100 * p))  # I got about 96%


if __name__ == '__main__':
    evaluate()
