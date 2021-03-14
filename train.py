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

from evaluation import users2itemids, hr_k, mrr_k
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
    return train_loss, sgns


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


def save_model(cnfg, model):
    ivectors = model.ivectors.weight.data.cpu().numpy()
    ovectors = model.ovectors.weight.data.cpu().numpy()
    pickle.dump(ivectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ivec.dat'), 'wb'))
    pickle.dump(ovectors, open(pathlib.Path(cnfg['save_dir'], 'idx2ovec.dat'), 'wb'))
    t.save(model, pathlib.Path(cnfg['save_dir'], 'best_model.pt'))


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
                               pathlib.Path(cnfg['data_dir'], cnfg['train']))
    for epoch in range(1, cnfg['max_epoch'] + 1):
        _train_loss = run_epoch(train_loader, epoch, sgns, optim)

    save_model(cnfg, model)


def evaluate(model, cnfg, user_lsts, eval_set):
    e_hr_k = hr_k(model, cnfg['k'], user_lsts, eval_set)
    # e_mrr_k = mrr_k(model, cnfg['k'], user_lsts, eval_set)
    # return e_hr_k * cnfg['hrk_weight'] + e_mrr_k * (1 - cnfg['hrk_weight'])
    return e_hr_k


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
                               pathlib.Path(cnfg['data_dir'], cnfg['train']))

    best_epoch = cnfg['max_epoch'] + 1
    valid_accs = [-np.inf]
    best_valid_acc = -np.inf
    train_losses = []
    patience_count = 0

    for epoch in range(1, cnfg['max_epoch'] + 1):
        train_loss, sgns = run_epoch(train_loader, epoch, sgns, optim)
        train_losses.append(train_loss)
        # TODO : send sgns instead of the model and reach to embedding.ivectors and embedding.ovectors
        valid_acc = evaluate(model, cnfg, user_lsts, eval_set)
        print(f'valid acc:{valid_acc}')

        diff_acc = valid_acc - valid_accs[-1]
        if diff_acc > cnfg['conv_thresh']:
            patience_count = 0
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch
                # TODO: seva sgns and not model ?
                save_model(cnfg, model)

        else:
            patience_count += 1
            if patience_count == cnfg['patience']:
                print(f"Early stopping")
                break

        valid_accs.append(valid_acc)

    if plot:
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(range(len(train_losses)), train_losses, label="train_loss")
        ax.plot(range(len(valid_accs)), valid_accs, label="valid_acc")
        ax.set_xlabel('epochs')

        ax.set_ylabel(r'train_loss')
        secaxy = ax.secondary_yaxis('right')
        secaxy.set_ylabel('valid_acc')

        plt.title('Train loss - Valid accuracy')
        # show a legend on the plot
        plt.legend()
        fig.savefig(f'plot_{str(cnfg["lr"])}.png')

    return best_epoch


def train_evaluate(cnfg):
    print(cnfg)
    user_lsts = users2itemids(pathlib.Path(cnfg['data_dir'], 'item2idx.dat'),
                            pathlib.Path(cnfg['data_dir'], 'vocab.dat'),
                            pathlib.Path(cnfg['data_dir'], 'train_corpus.txt'),
                            cnfg['unk'])
    eval_set = pd.read_csv(pathlib.Path(cnfg['data_dir'], 'valid.txt'))
    best_epoch = train_early_stop(cnfg, eval_set, user_lsts, plot=True)

    best_model = t.load(pathlib.Path(cnfg['save_dir'], 'best_model.pt'))

    acc = evaluate(best_model, cnfg, user_lsts, eval_set)
    return {'hr_k': (acc, 0.0), 'early_stop_epoch': (best_epoch, 0.0)}

