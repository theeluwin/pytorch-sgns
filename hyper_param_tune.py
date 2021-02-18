import argparse

from ax.service.managed_loop import optimize

from train import train_evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--max_epoch', type=int, default=100, help="max number of epochs")
    parser.add_argument('--k', type=int, default=10, help="number of top ranked items")
    parser.add_argument('--conv_thresh', type=float, default=0.0001, help="threshold diff for convergence")
    parser.add_argument('--patience', type=float, default=5, help="epochs to wait until early stopping")
    parser.add_argument('--hrk_weight', type=float, default=0.5, help="weight to put on hrk metric value")
    parser.add_argument('--trials', type=int, default=10, help="number of trials ")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")

    return parser.parse_args()


def full_train(cnfg, epochs, save_path):
    cnfg['valid'] = False
    cnfg['max_epoch'] = int(epochs)
    cnfg['save_dir'] = save_path
    _perf, _early_stop_epoch = train_evaluate(cnfg)


def main():
    args = parse_args()
    best_parameters, values, _experiment, _cur_model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "value_type": "float", "bounds": [1e-3, 1e-1], "log_scale": True},
            {"name": "ss_t", "type": "range", "value_type": "float", "bounds": [1e-5, 3e-3], "log_scale": True},
            {"name": "e_dim", "type": "choice", "value_type": "int", "values": [80, 100, 150, 200, 220, 250]},
            {"name": "n_negs", "type": "choice", "value_type": "int", "values": [5, 7, 10, 15, 50]},
            {"name": "mini_batch", "type": "choice", "value_type": "int", "values": [5, 8, 16, 36]},
            {"name": "weights", "type": "choice", "value_type": "bool", "values": [True, False]},
            {"name": "max_epoch", "type": "fixed", "value_type": "int", "value": args.max_epoch},
            {"name": "k", "type": "fixed", "value_type": "int", "value": args.k},
            {"name": "patience", "type": "fixed", "value_type": "int", "value": args.patience},
            {"name": "conv_thresh", "type": "fixed", "value_type": "float", "value": args.conv_thresh},
            {"name": "hrk_weight", "type": "fixed", "value_type": "float", "value": args.hrk_weight},
            {"name": "cuda", "type": "fixed", "value": args.cuda},
            {"name": "data_dir", "type": "fixed", "value_type": "str", "value": args.data_dir},

        ],
        evaluation_function=train_evaluate,
        minimize=False,
        objective_name='0.5*hr_k + 0.5*mrr_k',
        total_trials=args.trials
    )
    # full_train(best_parameters, values[0]['early_stop_epoch'], args.save_dir)


if __name__ == '__main__':
    main()
