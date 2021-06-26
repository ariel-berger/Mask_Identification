"""
Main file
We will run the whole program from here
"""

import torch
import hydra

from train import train
from dataset import MaskDataset
from models.base_model import MyModel
from utils import main_utils, train_utils
from torch.utils.data import DataLoader
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
from ray import tune
import numpy as np
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

torch.backends.cudnn.benchmark = True


# @hydra.main(config_path="config", config_name='config')
def main() -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    # Set seed for results reproduction
    # main_utils.init(cfg)
    # main_utils.set_seed(cfg['main']['seed'])

    # # Load dataset
    # train_dataset = MaskDataset(path=cfg['main']['paths']['train'])
    # # test_dataset = MaskDataset(path=cfg['main']['paths']['test'])
    #
    # train_size = int(0.8 * len(train_dataset))
    # validation_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])
    #
    # train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
    #                           num_workers=cfg['main']['num_workers'])
    # eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=False,
    #                          num_workers=cfg['main']['num_workers'])
    config = {}
    config["batch_size"] = 32
    config["dropout"] = tune.sample_from(lambda _:np.random.uniform(0,0.6))
    config["lr_value"] = tune.loguniform(1e-5, 1e-2)
    config["hidden_bb_dim"] = tune.sample_from(lambda _:np.random.randint(200,500))
    config["hidden_label_dim"] = tune.sample_from(lambda _:np.random.randint(200,500))
    config["bb_loss_weight"] = tune.sample_from(lambda _:np.random.randint(10,100))

    scheduler = ASHAScheduler(
        metric="score",
        mode="max",
        max_t=50,
        grace_period=2,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss","bb_loss","bce_loss", "iou_score", "accuracy_score", "training_iteration"])
    result = tune.run(
        train,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        config=config,
        num_samples=100,
        verbose=1,
        scheduler=scheduler,
       progress_reporter=reporter
    )


    # main_utils.init(cfg)
    # logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    # logger.write(OmegaConf.to_yaml(cfg))


    # test_loader = DataLoader(test_dataset, cfg['train']['batch_size'], shuffle=False,
    #                          num_workers=cfg['main']['num_workers'])

    # Init model
    # model = MyModel(dropout=cfg['train']['dropout'])
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #
    # if torch.cuda.is_available():
    #     model = model.cuda()

    # logger.write(main_utils.get_model_string(model))

    # Run model
    # train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    # analysis = tune.run(
    #     train, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})


    # metrics = train(model, train_loader, eval_loader, train_params, logger)
    # hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    # logger.report_metrics_hyper_params(hyper_parameters, metrics)


if __name__ == '__main__':
    main()
