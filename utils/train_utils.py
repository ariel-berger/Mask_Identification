"""
Includes all utils related to training
"""

import torch

from typing import Dict
from torch import Tensor
from utils.types import Metrics
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image


def get_zeroed_metrics_dict() -> Dict:
	"""
	:return: dictionary to store all relevant metrics for training
	"""
	return {'train_loss': 0, 'train_bb_loss': 0, 'train_bce_loss': 0, 'train_iou': 0, 'train_accuracy': 0,
			'total_norm': 0, 'count_norm': 0, 'bb_loss': 0, 'bce_loss': 0}


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
	"""
	return num of correct predictions
	:param pred: tensor of prediction
	:param labels: tensor of labels
	:return:
	"""
	output = (pred > 0.5)
	correct = 0
	for pred, label in zip(output, labels):
		if int(pred) == label:
			correct += 1
	return correct


def unscale_bb(bounding_box, shape):
	"""
	compute absolute bounding box
	:param bounding_box: [x, y, w, h] relative bounding box
	:param shape: shape of [x, y, w, h]
	:return:
	"""
	abs_x = bounding_box[:, 0] * shape[:, 0]
	abs_y = bounding_box[:, 1] * shape[:, 1]
	abs_width = bounding_box[:, 2] * shape[:, 0]
	abs_height = bounding_box[:, 3] * shape[:, 1]
	return [abs_x, abs_y, abs_width, abs_height]


def convert_to_xy(bounding_box):
	"""
	convert to x_min, x_max, y_min, y_max
	:param bounding_box:
	:return:
	"""
	x_min = bounding_box[0]
	x_max = bounding_box[0] + bounding_box[2]
	y_min = bounding_box[1]
	y_max = bounding_box[1] + bounding_box[3]
	return x_min, x_max, y_min, y_max


def calc_iou(predicted_bounding_box, bounding_box, shape):
	"""
	calculate iou
	:param predicted_bounding_box:
	:param bounding_box:
	:param shape:
	:return:
	"""
	x_min_hat, x_max_hat, y_min_hat, y_max_hat = convert_to_xy(unscale_bb(predicted_bounding_box, shape))
	x_min, x_max, y_min, y_max = convert_to_xy(unscale_bb(bounding_box, shape))
	inter_width = torch.maximum(torch.minimum(x_max, x_max_hat) - torch.maximum(x_min_hat, x_min),
								torch.zeros(x_min.size(0)).cuda())
	inter_height = torch.maximum(torch.minimum(y_max, y_max_hat) - torch.maximum(y_min_hat, y_min),
								 torch.zeros(x_min.size(0)).cuda())
	area_hat = (x_max_hat - x_min_hat) * (y_max_hat - y_min_hat)
	area_true = (x_max - x_min) * (y_max - y_min)
	area_intersection = inter_width * inter_height
	area_union = area_hat + area_true - area_intersection
	iou = area_intersection / area_union
	return iou.sum().item()


def print_img_bb(predicted_bounding_box, bounding_box, shape, path, txt):
	x_min_hat, x_max_hat, y_min_hat, y_max_hat = convert_to_xy(unscale_bb(predicted_bounding_box, shape))
	x_min, x_max, y_min, y_max = convert_to_xy(unscale_bb(bounding_box, shape))
	boxes = torch.tensor([[x_min_hat[0].item(), y_min_hat[0].item(), x_max_hat[0].item(), y_max_hat[0].item()],
						  [x_min[0].item(), y_min[0].item(), x_max[0].item(), y_max[0].item()]], dtype=torch.float)
	colors = ["red", "green"]
	img = read_image(path[0])
	result = draw_bounding_boxes(img, boxes, colors=colors, width=5)
	show(result, path[0], txt)


def show(imgs, path, txt):
	if not isinstance(imgs, list):
		imgs = [imgs]
	fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
	for i, img in enumerate(imgs):
		img = img.detach()
		img = F.to_pil_image(img)
		axs[0, i].imshow(np.asarray(img))
		axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
	img.save("./bb_images/" + txt + path[path.rfind('/', 1) + 1:])
