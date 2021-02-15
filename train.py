# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import Word2Vec, SGNS

from evaluation import users2items, hr_k, mrr_k
from config import DATA_DIR, VALID_PATH, TRAIN_PATH


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


def train(args):
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = ws if args.weights else None
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim)
    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights)
    if os.path.isfile(modelpath) and args.conti:
        sgns.load_state_dict(t.load(modelpath))
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters())
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))
    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(args.data_dir, 'train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))


def run_epoch(train_dl, epoch, sgns, optim):
    pbar = tqdm(train_dl)
    pbar.set_description("[Epoch {}]".format(epoch))
    for iword, owords in pbar:
        loss = sgns(iword, owords)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(train_loss=loss.item())


def evaluate(valid_dl, model):
    loss_lst = []
    pbar = tqdm(valid_dl)
    for iword, owords in pbar:
        loss = model(iword, owords)
        loss_lst.append(loss)

    return np.array(loss_lst).mean()


def train_to_dl(mini_batch_size):
    dataset = PermutedSubsampledCorpus(TRAIN_PATH)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
    total_batches = int(np.ceil(len(dataset) / mini_batch_size))
    print('total number of batches', total_batches)
    print('size of dataset', len(dataset))
    print('batches size with remain', len(dataset) / mini_batch_size)
    return dataloader


def train_evaluate(cnfg):
    idx2word = pickle.load(open(os.path.join(DATA_DIR, 'idx2word.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(DATA_DIR, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
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

#
# if __name__ == '__main__':
#     train(parse_args())
