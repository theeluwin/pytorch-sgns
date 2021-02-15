import pandas as pd
import random
from config import DATA_FPATH, TRAIN_PATH, VALID_PATH, DATA_COLS, POS_THRESH


def filter_group(group):
    ret_group = group[group['rating'] >= POS_THRESH]
    if ret_group.empty or ret_group.shape[0] < 2:
        print(f'user {group.name} rate less than 2 items above 3.5')
        return []
    else:
        return ret_group['item_id'].tolist()


def split_train_valid(lsts):
    with open(VALID_PATH, 'a') as valid_file, open(TRAIN_PATH, 'a') as train_file:
        valid_file.write('user_id,item_id\n')
        u = 1
        for u_lst in lsts:
            if len(u_lst):
                item = random.choice(u_lst)
                valid_file.write(str(u) + ',' + str(item) + '\n')
                u_lst.remove(item)
                out = ' '.join([str(i) for i in u_lst])
                train_file.write(out + '\n')
            else:
                train_file.write('' + '\n')
            u += 1


def main():
    data = pd.read_csv(DATA_FPATH, delimiter='\t', names=DATA_COLS)
    users2items = data.groupby('user_id').apply(lambda group: filter_group(group))
    split_train_valid(users2items)


if __name__ == '__main__':
    main()






