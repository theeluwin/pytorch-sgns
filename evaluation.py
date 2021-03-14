import codecs
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _extract_item_ids(corpus_line, vocab, item2idx, unk_token):
    return \
        [
            item2idx[item] if item in vocab else item2idx[unk_token]
            for item
            in corpus_line.split()
        ]


def users2itemids(item2index_path, vocab_path, corpus_path, unk_token):
    users = []
    item2idx = pickle.load(open(item2index_path, 'rb'))
    vocab = pickle.load(open(vocab_path, 'rb'))
    with codecs.open(corpus_path, 'r', encoding='utf-8') as corpus:
        for corpus_line in corpus:
            corpus_line = corpus_line.strip()

            if not corpus_line:
                users.append([])
                continue

            users.append(_extract_item_ids(corpus_line=corpus_line, vocab=vocab, item2idx=item2idx, unk_token=unk_token))
    return users


def represent_user(user_itemids, model):
    context_vecs = model.ivectors.weight.data.cpu().numpy()
    user2vec = context_vecs[user_itemids, :].mean(axis=0)
    return user2vec


def hr_k(model, k, users2itemids, eval_set, item2idx, unk):
    in_top_k = 0
    for u_id, target_item in eval_set[['user_id', 'item_id']].values:
        target_item_idx = item2idx.get(str(target_item), item2idx[unk])
        top_k_item_idxs = _calc_item_rank(k, model, u_id, users2itemids)
        if target_item_idx in top_k_item_idxs:
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


def _calc_item_rank(k, model, u_id, users2itemids):
    user2vec = represent_user(users2itemids[u_id], model).reshape(1, -1)
    user_sim = cosine_similarity(user2vec, model.ovectors.weight.data.cpu().numpy()).squeeze()
    top_k_items = user_sim.argsort()[-k:][::-1]
    return top_k_items

