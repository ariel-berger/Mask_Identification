"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn
from models.base_model import MyModel
from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from dataset import MaskDataset
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger
from ray import tune


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def accuracy(pred, labels):
    output = (pred > 0.5)
    correct = 0
    for pred, label in zip(output, labels):
        if int(pred) == label:
            correct += 1
    return correct


def unscale_bb(bounding_box, shape):
    rel_x = bounding_box[:, 0] * shape[:, 0]
    rel_y = bounding_box[:, 1] * shape[:, 1]
    rel_width = bounding_box[:, 2] * shape[:, 0]
    rel_height = bounding_box[:, 3] * shape[:, 1]
    return [rel_x, rel_y, rel_width, rel_height]


def convert_to_xy(bounding_box):
    x_min = bounding_box[0]
    x_max = bounding_box[0] + bounding_box[2]
    y_min = bounding_box[1]
    y_max = bounding_box[1] + bounding_box[3]
    return x_min, x_max, y_min, y_max


def calc_iou(predicted_bounding_box, bounding_box, shape):
    x_min_hat, x_max_hat, y_min_hat, y_max_hat = convert_to_xy(unscale_bb(predicted_bounding_box, shape))
    x_min, x_max, y_min, y_max = convert_to_xy(unscale_bb(bounding_box, shape))
    inter_width = torch.maximum(torch.minimum(x_max, x_max_hat) - torch.maximum(x_min_hat, x_min),
                                torch.zeros(x_min.size(0)).cuda())  # TODO - change to based on device
    inter_height = torch.maximum(torch.minimum(y_max, y_max_hat) - torch.maximum(y_min_hat, y_min),
                                 torch.zeros(x_min.size(0)).cuda())
    area_hat = (x_max_hat - x_min_hat) * (y_max_hat - y_min_hat)
    area_true = (x_max - x_min) * (y_max - y_min)
    area_intersection = inter_width * inter_height
    area_union = area_hat + area_true - area_intersection
    iou = area_intersection / area_union
    return iou.sum().item()


# def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
#           logger: TrainLogger) -> Metrics:
def train(config):
    model = MyModel(dropout=config['dropout'], hidden_bb_dim=config['hidden_bb_dim'],
                    hidden_label_dim=config['hidden_label_dim'])
    if torch.cuda.is_available():
        model = model.cuda()

    metrics = train_utils.get_zeroed_metrics_dict()
    best_score = 0

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_value'])

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)
    # Load dataset
    train_dataset = MaskDataset(path='/home/student/HW2/train')
    # test_dataset = MaskDataset(path=cfg['main']['paths']['test'])

    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=6)
    eval_loader = DataLoader(val_dataset, config['batch_size'], shuffle=False,
                             num_workers=6)

    for epoch in tqdm(range(30), maxinterval=100):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()

        for i, (image, bounding_box, label, shape) in enumerate(train_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                bounding_box = bounding_box.cuda()
                label = label.cuda()
                shape = shape.cuda()

            bb_hat, label_hat = model(image)
            bce_loss = nn.functional.binary_cross_entropy(label_hat.squeeze(-1), label)
            bb_loss = config["bb_loss_weight"] * nn.functional.smooth_l1_loss(bb_hat, bounding_box)
            loss = bce_loss + bb_loss

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            metrics['count_norm'] += 1
            metrics['train_accuracy'] += accuracy(label_hat, label)
            metrics['train_iou'] += calc_iou(bb_hat, bounding_box, shape)

            metrics['train_loss'] += loss.item() * image.size(0)
            metrics['train_bb_loss'] += bb_loss.item() * image.size(0)
            metrics['train_bce_loss'] += bce_loss.item() * image.size(0)

            # Report model to tensorboard
            # if epoch == 0 and i == 0:
            #     logger.report_graph(model, image)

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)
        metrics['train_bb_loss'] /= len(train_loader.dataset)
        metrics['train_bce_loss'] /= len(train_loader.dataset)

        metrics['train_accuracy'] /= len(train_loader.dataset)
        metrics['train_iou'] /= len(train_loader.dataset)
        metrics['train_accuracy'] *= 100
        metrics['train_iou'] *= 100
        iou_score, accuracy_score = metrics['train_iou'], metrics['train_accuracy']
        bce_loss, bb_loss, loss = metrics['train_bce_loss'], metrics['train_bb_loss'], metrics['train_loss']
        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_accuracy'], metrics['eval_iou'], metrics['eval_loss'] = evaluate(model, eval_loader)
        model.train(True)

        epoch_time = time.time() - t
        score = (metrics['eval_iou'] + metrics['eval_accuracy']) / 2
        tune.report(score=score, loss=metrics['eval_loss'], accuracy=metrics['eval_accuracy'], iou=metrics['eval_iou'],
                    train_bce_loss=bce_loss, train_bb_loss=bb_loss, train_loss=loss, train_accuracy=accuracy_score,
                    train_iou=iou_score)

        # self, epoch: int, epoch_time: float, train_loss: float, train_bce_loss: float,
        # train_l1_loss: float, norm: float,
        # train_accuracy: float, train_iou: float, eval_accuracy: float, eval_iou: float
        # , eval_loss
        # logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'],metrics['train_bce_loss'],
        #                               metrics['train_bb_loss'],norm, metrics['train_accuracy'], metrics['train_iou'],
        #                               metrics['eval_accuracy'], metrics['eval_iou'], metrics['eval_loss'])

        # scalars = {'Accuracy/Train': metrics['train_accuracy'],
        #            'iou/Train': metrics['train_iou'],
        #            'Accuracy/Validation': metrics['eval_accuracy'],
        #            'iou/Validation': metrics['eval_iou'],
        #            'Loss/Train': metrics['train_loss'],
        #            'Loss bb/Train': metrics['train_bb_loss'],
        #            'Loss cbe/Train': metrics['train_bce_loss'],
        #            'Loss/Validation': metrics['eval_loss']}

        # logger.report_scalars(scalars, epoch)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "./model.pth")

    return get_metrics(best_score, metrics['eval_accuracy'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    accuracy_score = 0
    iou_score = 0
    loss = 0

    for i, (image, bounding_box, label, shape) in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            image = image.cuda()
            bounding_box = bounding_box.cuda()
            label = label.cuda()
            shape = shape.cuda()

        bb_hat, label_hat = model(image)

        bce_loss = nn.functional.binary_cross_entropy(label_hat.squeeze(-1), label)
        bb_loss = 10 * nn.functional.smooth_l1_loss(bb_hat, bounding_box)
        loss += bce_loss + bb_loss
        accuracy_score += accuracy(label_hat, label)
        iou_score += calc_iou(bb_hat, bounding_box, shape)

    loss /= len(dataloader.dataset)
    accuracy_score /= len(dataloader.dataset)
    iou_score /= len(dataloader.dataset)
    accuracy_score *= 100
    iou_score *= 100

    return accuracy_score, iou_score, loss.item()
