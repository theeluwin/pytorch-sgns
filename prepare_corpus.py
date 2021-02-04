import pandas as pd

from config import DATA_FPATH, CORPUS_FPATH, DATA_COLS, POS_THRESH


def filter_positives(df):
    return df[df['rating'] >= POS_THRESH]


def build_u_lsts(df):
    return df.sort_values('timestamp').groupby('user_id').item_id.apply(list).tolist()


def extract_corpus(u_lsts):
    with open(CORPUS_FPATH, 'a') as the_file:
        for u in u_lsts:
            u = [str(i) for i in u]
            sent = ' '.join(u)
            the_file.write(sent + '\n')


def prepare_corpus():
    data = pd.read_csv(DATA_FPATH, delimiter='\t', names=DATA_COLS)
    data = filter_positives(data)
    u_lsts = build_u_lsts(data)
    extract_corpus(u_lsts)

# if __name__ == '__main__':
#     data = pd.read_csv(DATA_FPATH, delimiter='\t', names=DATA_COLS)
#     data = filter_positives(data)
#     u_lsts = build_u_lsts(data)
#     extract_corpus(u_lsts)