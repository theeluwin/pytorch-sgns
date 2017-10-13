# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn

from torch import LongTensor
from torch.autograd import Variable


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, V, d=50, padding_idx=0, use_gpu=False):
        super(Word2Vec, self).__init__()
        self.V = V + 1
        self.d = d
        self.use_gpu = use_gpu
        self.ivectors = nn.Embedding(self.V, self.d, padding_idx=padding_idx, sparse=True)
        self.ovectors = nn.Embedding(self.V, self.d, padding_idx=padding_idx, sparse=True)
        if self.use_gpu:
            self.ivectors = self.ivectors.cuda()
            self.ovectors = self.ovectors.cuda()

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = Variable(LongTensor(data))
        if self.use_gpu:
            v = v.cuda()
        return self.ivectors(v)

    def forward_o(self, data):
        v = Variable(LongTensor(data))
        if self.use_gpu:
            v = v.cuda()
        return self.ovectors(v)


class SGNS(nn.Module):

    def __init__(self, V, embedding, batch_size=128, window_size=4, n_negatives=5):
        super(SGNS, self).__init__()
        self.V = V + 1
        self.embedding = embedding
        self.batch_size = batch_size
        self.window_size = window_size
        self.n_negatives = n_negatives

    def sample(self, iword_b, owords_b):
        nwords_b = []
        for b in range(self.batch_size):
            iword = iword_b[b]
            owords = owords_b[b]
            nwords = []
            for oword in owords:
                negs = []
                while True:
                    if len(negs) >= self.n_negatives:
                        break
                    idx = random.randrange(1, self.V)
                    if (idx == iword) or (idx in owords) or (idx in negs):
                        continue
                    negs.append(idx)
                nwords.append(negs)
            nwords_b.append(nwords)
        return nwords_b

    def forward(self, iword, owords):
        # black magic from https://github.com/kefirski/pytorch_NEG_loss
        nwords = self.sample(iword, owords)
        ivectors = self.embedding.forward_i(LongTensor(iword).repeat(1, self.window_size).contiguous().view(-1))
        ovectors = self.embedding.forward_o(LongTensor(owords).contiguous().view(-1))
        nvectors = self.embedding.forward_o(LongTensor(nwords).contiguous().view(self.batch_size * self.window_size, -1)).neg()
        oloss = (ivectors * ovectors).sum(1).squeeze().sigmoid().log()
        nloss = torch.bmm(nvectors, ivectors.unsqueeze(2)).sigmoid().log().sum(1).squeeze()
        return -(oloss + nloss).sum() / self.batch_size
