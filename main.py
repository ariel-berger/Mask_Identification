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
import json
from train_separately import train as train_separately

torch.backends.cudnn.benchmark = True


def main() -> None:
	# Setting hyperparameters for training
	config = {}
	config["batch_size"] = 16
	config["dropout"] = 0.098203#tune.sample_from(lambda _: np.random.uniform(0, 0.6))
	config["lr_value"] = 0.00034058#tune.loguniform(1e-5, 1e-2)
	config["hidden_bb_dim"] = 298#tune.sample_from(lambda _: np.random.randint(200, 500))
	config["hidden_label_dim"] = 259#tune.sample_from(lambda _: np.random.randint(200, 500))
	config["bb_loss_weight"] = 28#tune.sample_from(lambda _: np.random.randint(10, 100))
	config["step_size"] = 30
	config["train_path"] ='/home/student/HW2/train'

	# tuning model using ray tune, updating tensorboard
	scheduler = ASHAScheduler(
		metric="score",
		mode="max",
		max_t=50,
		grace_period=2,
		reduction_factor=2)
	reporter = CLIReporter(
		metric_columns=["loss", "bb_loss", "bce_loss", "iou_score", "accuracy_score", "training_iteration"])
	# train(config)
	result = tune.run(
		train_separately,
		resources_per_trial={"cpu": 2, "gpu": 1},
		config=config,
		num_samples=1,
		verbose=1,
		scheduler=scheduler,
		progress_reporter=reporter
	)


if __name__ == '__main__':
	main()
