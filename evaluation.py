import codecs
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = 'data/'
CORPUS_PATH = DATA_DIR + 'corpus.txt'
VOCAB_PATH = DATA_DIR + 'vocab.dat'
ITEM2IDX_PATH = DATA_DIR + 'word2idx.dat'
VALID_PATH = DATA_DIR + 'valid.txt'
UNK = '<UNK>'


def _extract_user_items(corpus_line, vocab, item2idx):
    return \
        [
            item2idx[item] if item in vocab else item2idx[UNK]
            for item
            in corpus_line.split()
        ]


def users2items():
    users = []
    item2idx = pickle.load(open(ITEM2IDX_PATH, 'rb'))
    vocab = pickle.load(open(VOCAB_PATH, 'rb'))
    with codecs.open(CORPUS_PATH, 'r', encoding='utf-8') as corpus:
        for corpus_line in corpus:
            corpus_line = corpus_line.strip()

            if not corpus_line:
                users.append([])
                continue

            users.append(_extract_user_items(corpus_line=corpus_line, vocab=vocab, item2idx=item2idx))
    return users


def represent_user(user_items, model):
    """
    represent each user as the mean of his items context representations.
    :param user_items: list of item indices of a specific user.
    :param model: SGNS model.
    :return: user representation
    """
    context_vecs = model.ovectors.weight.data.cpu().numpy()
    user2vec = context_vecs[user_items, :].mean(axis=0)
    return user2vec


def hr_k(model, k, users2items, eval_set):
    in_top_k = 0
    for u_id, target_item in eval_set[['user_id', 'item_id']].values:
        # TODO convert it to tensors?
        top_k_items = _calc_item_rank(k, model, u_id, users2items)
        if target_item in top_k_items:
            in_top_k += 1
    hr_k = in_top_k / eval_set.shape[0]
    return hr_k


def mrr_k(model, k, users2items, eval_set):
    in_top_k, rec_rank = 0, 0
    for u_id, target_item in eval_set[['user_id', 'item_id']].values:
        top_k_items = _calc_item_rank(k, model, u_id, users2items)
        if target_item in top_k_items:
            in_top_k += 1
            rec_rank += 1 / (np.where(top_k_items == target_item)[0][0] + 1)
    mrp_k = rec_rank / in_top_k
    return mrp_k


def _calc_item_rank(k, model, u_id, users2items):
    user2vec = represent_user(users2items[u_id - 1], model).reshape(1, -1)
    user_sim = cosine_similarity(user2vec, model.ivectors.weight.data.cpu().numpy()).squeeze()
    top_k_items = user_sim.argsort()[-k:][::-1] + 1
    return top_k_items

