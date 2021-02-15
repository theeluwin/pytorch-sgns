# -*- coding: utf-8 -*-

import pathlib
import pickle
import random

import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import DATA_DIR, VALID_PATH, TRAIN_PATH
from evaluation import users2items, hr_k, mrr_k
from model import Word2Vec, SGNS


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def run_epoch(train_dl, epoch, sgns, optim):
    pbar = tqdm(train_dl)
    pbar.set_description("[Epoch {}]".format(epoch))

    for iword, owords in pbar:
        loss = sgns(iword, owords)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(train_loss=loss.item())


def train_to_dl(mini_batch_size):
    dataset = PermutedSubsampledCorpus(TRAIN_PATH)
    return DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)


def train_evaluate(cnfg):
    idx2word = pickle.load(pathlib.Path(DATA_DIR, 'idx2word.dat').open('rb'))
    wc = pickle.load(pathlib.Path(DATA_DIR, 'wc.dat').open('rb'))

    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()

    assert (wf > 0).all(), 'Items with invalid count appear.'
    ws = 1 - np.sqrt(cnfg['ss_t'] / wf)
    ws = np.clip(ws, 0, 1)

    vocab_size = len(idx2word)
    weights = ws if cnfg['weights'] else None
    model = Word2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])

    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adam(sgns.parameters(), lr=cnfg['lr'])

    train_loader = train_to_dl(cnfg['mini_batch'])
    user_lsts = users2items()
    eval_set = pd.read_csv(VALID_PATH)

    last_epoch_perf = -np.inf
    for epoch in range(1, cnfg['max_epoch'] + 1):
        run_epoch(train_loader, epoch, sgns, optim)
        e_hr_k = hr_k(model, cnfg['k'], user_lsts, eval_set)
        e_mrr_k = mrr_k(model, cnfg['k'], user_lsts, eval_set)
        perf = e_hr_k * cnfg['hrk_weight'] + e_mrr_k * (1 - cnfg['hrk_weight'])
        perf_diff = perf - last_epoch_perf
        if perf_diff < cnfg['conv_thresh']:
            print(f"Early stop at epoch:{epoch}")
            print(f"HR at {cnfg['k']}:{e_hr_k}, MRR at {cnfg['k']}:{e_mrr_k}")
            break

        last_epoch_perf = perf

    return last_epoch_perf

