import pickle
import codecs
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = 'data/'
CORPUS_PATH = DATA_DIR + 'corpus.txt'
VOCAB_PATH = DATA_DIR + 'vocab.dat'
ITEM2IDX_PATH = DATA_DIR + 'word2idx.dat'
VALID_PATH = DATA_DIR + 'valid.txt'
UNK = '<UNK>'


def evaluate(model, k):
    item2idx = pickle.load(open(ITEM2IDX_PATH, 'rb'))
    vocab = pickle.load(open(VOCAB_PATH, 'rb'))
    with codecs.open(CORPUS_PATH, 'r', encoding='utf-8') as corpus, codecs.open(VALID_PATH, 'r', encoding='utf-8') as valid:
        u = 1
        in_top_k, rec_rank = 0, 0
        for corpus_line, valid_line in zip(corpus, valid):
            corpus_line = corpus_line.strip()
            if not corpus_line:
                print(f'user {u} is not represented, thus not evaluated')
                continue
            user = []
            for item in corpus_line.split():
                if item in vocab:
                    user.append(item)
                else:
                    user.append(UNK)
            user = [item2idx[item] for item in user]
            context_vecs = model.ovectors.weight.data.cpu().numpy()
            user2vec = context_vecs[user, :].mean(axis=0)
            user_sim = cosine_similarity(user2vec, model.ivectors.weight.data.cpu().numpy())
            target_item = int(valid_line)
            top_k_items = user_sim.argsort()[-k:][::-1] + 1
            if target_item in top_k_items:
                in_top_k += 1
                rec_rank += 1 / (np.where(top_k_items == target_item)[0][0] + 1)
            u += 1
    hr_k = in_top_k / (u -1)
    mrp_k = rec_rank / in_top_k
    return hr_k, mrp_k
