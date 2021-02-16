import random
import argparse

import pandas as pd

DATA_COLS = ['user_id', 'item_id', 'rating', 'timestamp']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--source_data', type=str, default='./ml-100k/u.data', help="source data of user-item rankings")
    parser.add_argument('--corpus_path', type=str, default='./data/corpus.txt', help="path to save corpus")
    parser.add_argument('--valid_path', type=str, default='./data/valid.txt', help="path to save validation")
    parser.add_argument('--pos_thresh', type=float, default=3.5, help="rank threshold to assign for positive items")
    parser.add_argument('--min_items', type=int, default=2, help="min number of positive items needed to store a user")
    return parser.parse_args()


def filter_group(group, pos_thresh, min_items):
    ret_group = group[group['rating'] >= pos_thresh]
    if ret_group.empty or ret_group.shape[0] < min_items:
        print(f'user {group.name} ranked less than {min_items} items above {pos_thresh}')
        return []
    else:
        return ret_group['item_id'].tolist()


def split_train_valid(lsts, corpus_path, valid_path):
    with open(corpus_path, 'a') as corpus_file, open(valid_path, 'a') as valid_file:
        valid_file.write('user_id,item_id\n')
        for u in range(lsts.shape[0]):
            u_lst = lsts[u]
            if len(u_lst):
                item = random.choice(u_lst)
                valid_file.write(str(u) + ',' + str(item) + '\n')
                u_lst.remove(item)
                out = ' '.join([str(i) for i in u_lst])
                corpus_file.write(out + '\n')
            else:
                corpus_file.write('' + '\n')


def read_data(path, data_cols):
    data = pd.read_csv(path, delimiter='\t', names=data_cols)
    data[['user_id', 'item_id']] = data[['user_id', 'item_id']].apply(lambda col: col-1)
    return data


def main():
    args = parse_args()
    data = read_data(args.source_data, DATA_COLS)
    users2items = data.groupby('user_id').apply(lambda group: filter_group(group, args.pos_thresh, args.min_items))
    print(f'number of users: {len(users2items)}, number of items: '
          f'{len(set([items for item in users2items.tolist() for items in item]))}')
    split_train_valid(users2items, args.corpus_path, args.valid_path)


if __name__ == '__main__':
    main()






