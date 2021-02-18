# -*- coding: utf-8 -*-

import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import torch as t
from torch.optim import Adagrad
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from evaluation import users2items, hr_k, mrr_k
from model import Item2Vec, SGNS

import matplotlib.pyplot as plt


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
    train_losses = []

    for iitem, oitems in pbar:
        loss = sgns(iitem, oitems)
        train_losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(train_loss=loss.item())

    train_loss = np.array(train_losses).mean()
    print(f'train_loss: {train_loss}')
    return train_loss


def train_to_dl(mini_batch_size, train_path):
    dataset = PermutedSubsampledCorpus(train_path)
    return DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)


def configure_weights(cnfg, idx2item):
    ic = pickle.load(pathlib.Path(cnfg['data_dir'], 'ic.dat').open('rb'))

    ifr = np.array([ic[item] for item in idx2item])
    ifr = ifr / ifr.sum()

    assert (ifr > 0).all(), 'Items with invalid count appear.'
    istt = 1 - np.sqrt(cnfg['ss_t'] / ifr)
    istt = np.clip(istt, 0, 1)
    weights = istt if cnfg['weights'] else None
    return weights


def train(cnfg):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = Item2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])

    train_loader = train_to_dl(cnfg['mini_batch'],
                               pathlib.Path(cnfg['data_dir'], 'train.dat'))
    for epoch in range(1, cnfg['max_epoch'] + 1):
        _train_loss = run_epoch(train_loader, epoch, sgns, optim)

    ivectors = model.ivectors.weight.data.cpu().numpy()
    ovectors = model.ovectors.weight.data.cpu().numpy()
    pickle.dump(ivectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ivec.dat'), 'wb'))
    pickle.dump(ovectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ovec.dat'), 'wb'))
    t.save(sgns.state_dict(), pathlib.Path(cnfg['save_dir'], 'sgns.pt'))


def evaluate(model, cnfg, user_lsts, eval_set):
    e_hr_k = hr_k(model, cnfg['k'], user_lsts, eval_set)
    e_mrr_k = mrr_k(model, cnfg['k'], user_lsts, eval_set)
    return e_hr_k * cnfg['hrk_weight'] + e_mrr_k * (1 - cnfg['hrk_weight'])


def train_early_stop(cnfg, eval_set, user_lsts, plot=True):
    idx2item = pickle.load(pathlib.Path(cnfg['data_dir'], 'idx2item.dat').open('rb'))

    weights = configure_weights(cnfg, idx2item)
    vocab_size = len(idx2item)

    model = Item2Vec(vocab_size=vocab_size, embedding_size=cnfg['e_dim'])
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=cnfg['n_negs'], weights=weights)
    if cnfg['cuda']:
        sgns = sgns.cuda()

    optim = Adagrad(sgns.parameters(), lr=cnfg['lr'])

    train_loader = train_to_dl(cnfg['mini_batch'],
                               pathlib.Path(cnfg['data_dir'], 'train.dat'))

    early_stop_epoch = cnfg['max_epoch'] + 1
    valid_accs = []
    train_losses = []
    patience_count = 0
    best_model = None

    for epoch in range(1, cnfg['max_epoch'] + 1):
        train_loss = run_epoch(train_loader, epoch, sgns, optim)
        train_losses.append(train_loss)
        valid_acc = evaluate(model, cnfg, user_lsts, eval_set)
        print(f'valid acc:{valid_acc}')

        patience_count += int(valid_acc - valid_accs[-1] < cnfg['conv_thresh'])
        if patience_count == cnfg['patience']:
            early_stop_epoch = epoch
            print(f"Early stop at epoch:{epoch}")
            break

        if valid_acc > valid_accs[-1]:
            best_model = model
        valid_accs.append(valid_acc)

    if plot:
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(range(len(train_losses)), train_losses, label="train_loss")
        ax.plot(range(len(valid_accs)), valid_accs, label="valid_acc")
        ax.xlabel('epochs')

        ax.set_ylabel(r'train_loss')
        secaxx = ax.secondary_xaxis('top')
        secaxx.set_xlabel('valid_acc')

        plt.title('Train loss - Valid accuracy')
        # show a legend on the plot
        plt.legend()
        fig.savefig(f'plot_{str(cnfg["lr"])}')

    return best_model, early_stop_epoch


def train_evaluate(cnfg):
    user_lsts = users2items(pathlib.Path(cnfg['data_dir'], 'item2idx.dat'),
                            pathlib.Path(cnfg['data_dir'], 'vocab.dat'),
                            pathlib.Path(cnfg['data_dir'], 'train_corpus.txt'))
    eval_set = pd.read_csv(pathlib.Path(cnfg['data_dir'], 'valid.txt'))
    best_model, early_stop_epoch = train_early_stop(cnfg, eval_set, user_lsts, plot=True)

    acc = evaluate(best_model, cnfg, user_lsts, eval_set)
    return {'0.5*hr_k + 0.5*mrr_k': (acc, 0.0), 'early_stop_epoch': (early_stop_epoch, 0.0)}

