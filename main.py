"""
Main file
We will run the whole program from here
"""

import torch
from train import train
from ray import tune
import numpy as np
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from train_separately import train as train_separately
torch.backends.cudnn.benchmark = True


def main() -> None:
    # Setting hyperparameters for training
    config = {}
    config["batch_size"] = 16
    config["dropout"] = tune.sample_from(lambda _: np.random.uniform(0, 0.6))
    config["lr_value"] = tune.loguniform(1e-5, 1e-2)
    config["hidden_bb_dim"] = tune.sample_from(lambda _: np.random.randint(200, 500))
    config["hidden_label_dim"] = tune.sample_from(lambda _: np.random.randint(200, 500))
    config["bb_loss_weight"] = tune.sample_from(lambda _: np.random.randint(10, 100))

    # tuning model using ray tune, updating tensorboard
    scheduler = ASHAScheduler(
        metric="score",
        mode="max",
        max_t=50,
        grace_period=2,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "bb_loss", "bce_loss", "iou_score", "accuracy_score", "training_iteration"])

    result = tune.run(
        train_separately,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=100,
        verbose=1,
        scheduler=scheduler,
        progress_reporter=reporter
    )


if __name__ == '__main__':
    main()
