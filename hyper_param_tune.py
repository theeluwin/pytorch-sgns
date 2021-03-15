import argparse

from ax.service.managed_loop import optimize

from train import train_evaluate, train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./output/', help="model directory path")
    parser.add_argument('--train', type=str, default='train.dat', help="train file name")
    parser.add_argument('--full_train', type=str, default='full_train.dat', help="full train file name")
    parser.add_argument('--max_epoch', type=int, default=100, help="max number of epochs")
    parser.add_argument('--k', type=int, default=20, help="number of top ranked items")
    parser.add_argument('--conv_thresh', type=float, default=0.0001, help="threshold diff for convergence")
    parser.add_argument('--patience', type=float, default=3, help="epochs to wait until early stopping")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--hrk_weight', type=float, default=0.5, help="weight to put on hrk metric value")
    parser.add_argument('--trials', type=int, default=10, help="number of trials ")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")

    return parser.parse_args()


def full_train(cnfg, epochs, args):
    cnfg['max_epoch'] = int(epochs)
    cnfg['train'] = args.full_train
    train(cnfg)


def main():
    args = parse_args()
    best_parameters, values, _experiment, _cur_model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "value_type": "float", "bounds": [5e-2, 3e-1]},
            {"name": "ss_t", "type": "range", "value_type": "float", "bounds": [1e-5, 3e-3]},
            {"name": "e_dim", "type": "choice", "value_type": "int", "values": [12, 17, 20, 25, 30]},
            {"name": "n_negs", "type": "choice", "value_type": "int", "values": [5, 7, 10, 15]},
            {"name": "mini_batch", "type": "choice", "value_type": "int", "values": [100, 90]},
            {"name": "weights", "type": "choice", "value_type": "bool", "values": [False, False]},
            {"name": "max_epoch", "type": "fixed", "value_type": "int", "value": args.max_epoch},
            {"name": "k", "type": "fixed", "value_type": "int", "value": args.k},
            {"name": "patience", "type": "fixed", "value_type": "int", "value": args.patience},
            {"name": "conv_thresh", "type": "fixed", "value_type": "float", "value": args.conv_thresh},
            {"name": "hrk_weight", "type": "fixed", "value_type": "float", "value": args.hrk_weight},
            {"name": "unk", "type": "fixed", "value_type": "str", "value": args.unk},
            {"name": "cuda", "type": "fixed", "value": args.cuda},
            {"name": "data_dir", "type": "fixed", "value_type": "str", "value": args.data_dir},
            {"name": "save_dir", "type": "fixed", "value_type": "str", "value": args.save_dir},
            {"name": "train", "type": "fixed", "value_type": "str", "value": args.train}
        ],
        evaluation_function=train_evaluate,
        minimize=False,
        objective_name='hr_k',
        total_trials=args.trials
    )

    full_train(best_parameters, values[0]['early_stop_epoch'], args)


if __name__ == '__main__':
    main()
