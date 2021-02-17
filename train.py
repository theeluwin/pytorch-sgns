# -*- coding: utf-8 -*-

import pathlib
import pickle
import random

import numpy as np
import pandas as pd
from torch.optim import Adagrad
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from evaluation import users2items, hr_k, mrr_k
from model import Item2Vec, SGNS


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(datapath.open('rb'))
        if ws is not None:
            self.data = []
            for iitem, oitems in data:
                if random.random() > ws[iitem]:
                    self.data.append((iitem, oitems))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iitem, oitems = self.data[idx]
        return iitem, np.array(oitems)


def run_epoch(train_dl, epoch, sgns, optim):
    pbar = tqdm(train_dl)
    pbar.set_description("[Epoch {}]".format(epoch))

    for iitem, oitems in pbar:
        loss = sgns(iitem, oitems)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(train_loss=loss.item())

    return loss.item()


def train_to_dl(mini_batch_size, train_path):
    dataset = PermutedSubsampledCorpus(train_path)
    return DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)


def train_evaluate(cnfg):
    print(cnfg)
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))
    ic = pickle.load(pathlib.Path(cnfg['data_dir'], 'ic.dat').open('rb'))

    ifr = np.array([ic[item] for item in idx2item])
    ifr = ifr / ifr.sum()

    assert (ifr > 0).all(), 'Items with invalid count appear.'
    istt = 1 - np.sqrt(cnfg['ss_t'] / ifr)
    istt = np.clip(istt, 0, 1)

    vocab_size = len(idx2item)
    weights = istt if cnfg['weights'] else None
    model = Item2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])

    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])

    train_loader = train_to_dl(cnfg['mini_batch'],
                               pathlib.Path(cnfg['data_dir'], 'train.dat'))

    last_epoch_perf = -np.inf
    early_stop_epoch = cnfg['max_epoch'] + 1

    # early_stopping = EarlyStopping(
    #     monitor='val_accuracy',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='max'
    # )
    for epoch in range(1, cnfg['max_epoch'] + 1):
        train_loss = run_epoch(train_loader, epoch, sgns, optim)
        if cnfg['valid']:
            user_lsts = users2items(pathlib.Path(cnfg['data_dir'], 'item2idx.dat'),
                                    pathlib.Path(cnfg['data_dir'], 'vocab.dat'),
                                    pathlib.Path(cnfg['data_dir'], 'train_corpus.txt'))
            eval_set = pd.read_csv(pathlib.Path(cnfg['data_dir'], 'valid.txt'))
            e_hr_k = hr_k(model, cnfg['k'], user_lsts, eval_set)
            e_mrr_k = mrr_k(model, cnfg['k'], user_lsts, eval_set)
            perf = e_hr_k * cnfg['hrk_weight'] + e_mrr_k * (1 - cnfg['hrk_weight'])
            perf_diff = perf - last_epoch_perf

            # # early_stopping needs the validation loss to check if it has decresed,
            # # and if it has, it will make a checkpoint of the current model
            # early_stopping(perf, model)
            #
            # if early_stopping.early_stop:
            #     early_stop_epoch = epoch
            #     print("Early stopping")
            #     break
            print(f'train_loss:{train_loss}, valid acc:{perf}')
            if perf_diff < cnfg['conv_thresh']:
                early_stop_epoch = epoch
                print(f"Early stop at epoch:{epoch}")
                print(f"HR at {cnfg['k']}:{e_hr_k}, MRR at {cnfg['k']}:{e_mrr_k}")
                break

            last_epoch_perf = perf
        else:
            print(f'train_loss:{train_loss}')

    return {'0.5*hr_k + 0.5*mrr_k': (last_epoch_perf, 0.0), 'early_stop_epoch': (early_stop_epoch, 0.0)}

