# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS


class PermutedCorpus(Dataset):

    def __init__(self, datapath):
        self.data = pickle.load(open(datapath, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(gpu=False):
    num_epochs = 2
    batch_size = 256
    every = 10
    vocab = pickle.load(open('./data/vocab.dat', 'rb'))
    V = len(vocab)
    word2vec = Word2Vec(V=V, gpu=gpu)
    sgns = SGNS(V=V, embedding=word2vec, batch_size=batch_size, window_size=4, n_negatives=5)
    optimizer = SGD(sgns.parameters(), 5e-1)
    dataset = PermutedCorpus('./data/train.dat')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    start = time.time()
    total_batches = len(dataset) // batch_size
    for epoch in range(1, num_epochs + 1):
        for batch, (iword, owords) in enumerate(dataloader):
            if len(iword) != batch_size:
                continue
            loss = sgns(iword, owords)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not batch % every:
                print("[e{}][b{}/{}] loss: {:7.4f}\r".format(epoch, batch, total_batches, loss.data[0]))
    end = time.time()
    print("training done in {:.4f} seconds".format(end - start))  # It takes about 3.5 minutes with GPU, loss less than 7.5
    idx2vec = word2vec.forward([idx for idx in range(V + 1)])
    if gpu:
        idx2vec = idx2vec.cpu()
    pickle.dump(word2vec.state_dict(), open('./data/word2vec.pt', 'wb'))
    pickle.dump(idx2vec.data.numpy(), open('./data/idx2vec.dat', 'wb'))


if __name__ == '__main__':
    train()
