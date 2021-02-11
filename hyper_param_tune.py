from ax.service.managed_loop import optimize
from train import train_evaluate

NUM_OF_TRIALS = 10
best_parameters, values, experiment, cur_model = optimize(
    parameters=[
        {"name": "lr", "type": "choice", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "ss_t", "type": "choice", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "e_dim", "type": "choice", "values": [80, 100, 150, 200, 220, 250]},
        {"name": "n_negs", "type": "choice", "values": [5, 7, 10, 15, 50]},
        {"name": "mini_batch", "type": "choice", "bounds": [5, 50]},
        {"name": "weights", "type": "choice", "values": [True, False]}
    ],
    evaluation_function=train_evaluate,
    minimize=True,
    objective_name='0.5*hr_k + 0.5*mrr_k',
    total_trials=NUM_OF_TRIALS
)

