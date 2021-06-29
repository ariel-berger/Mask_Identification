import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from dataset import MaskDataset
from models.separate_model import MaskModel, BBModel
from utils.train_utils import get_metrics, accuracy, calc_iou, print_img_bb, unscale_bb
import numpy as np

from ray import tune


def train(config):
	bb_model = BBModel(dropout=config['dropout'], hidden_bb_dim=config['hidden_bb_dim'])
	mask_model = MaskModel(dropout=config['dropout'], hidden_label_dim=config['hidden_label_dim'])

	if torch.cuda.is_available():
		bb_model = bb_model.cuda()
		mask_model = mask_model.cuda()

	best_accuracy = 0
	best_iou = 0

	# Create optimizer
	optimizer = torch.optim.Adam(list(bb_model.parameters()) + list(mask_model.parameters()), lr=config['lr_value'])

	# Create learning rate scheduler
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'])
	# Load dataset
	train_dataset = MaskDataset(path=config['train_path'])

	train_size = int(0.8 * len(train_dataset))
	validation_size = len(train_dataset) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])

	train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
							  num_workers=6)
	eval_loader = DataLoader(val_dataset, config['batch_size'], shuffle=False,
							 num_workers=6)

	for epoch in tqdm(range(25), maxinterval=100):
		t = time.time()
		metrics = train_utils.get_zeroed_metrics_dict()

		for i, (image, bounding_box, label, shape, path) in enumerate(train_loader):
			if torch.cuda.is_available():
				image = image.cuda()
				bounding_box = bounding_box.cuda()
				label = label.cuda()
				shape = shape.cuda()

			bb_hat = bb_model(image)
			label_hat = mask_model(image)
			# compute losses
			bce_loss = nn.functional.binary_cross_entropy(label_hat.squeeze(-1), label)
			bb_loss = nn.functional.smooth_l1_loss(bb_hat, bounding_box)

			# Optimization step
			optimizer.zero_grad()
			bce_loss.backward()
			bb_loss.backward()
			optimizer.step()

			# Calculate metrics
			metrics['train_accuracy'] += accuracy(label_hat, label)
			metrics['train_iou'] += calc_iou(bb_hat, bounding_box, shape)

			metrics['train_bb_loss'] += bb_loss.item() * image.size(0)
			metrics['train_bce_loss'] += bce_loss.item() * image.size(0)

		# Learning rate scheduler step
		scheduler.step()

		metrics['train_bb_loss'] /= len(train_loader.dataset)
		metrics['train_bce_loss'] /= len(train_loader.dataset)
		metrics['train_accuracy'] /= len(train_loader.dataset)
		metrics['train_iou'] /= len(train_loader.dataset)
		metrics['train_accuracy'] *= 100
		metrics['train_iou'] *= 100
		train_iou, train_accuracy = metrics['train_iou'], metrics['train_accuracy']
		bce_loss, bb_loss = metrics['train_bce_loss'], metrics['train_bb_loss']

		# run evaluation on validation set
		bb_model.train(False)
		mask_model.train(False)
		metrics['eval_accuracy'], metrics['eval_iou'], metrics['bce_loss'], metrics['bb_loss'] = evaluate(bb_model,
																										  mask_model,
																										  eval_loader)
		bb_model.train(True)
		mask_model.train(True)

		score = (metrics['eval_accuracy'] + metrics['eval_iou']) / 2
		tune.report(score=score, accuracy=metrics['eval_accuracy'], iou=metrics['eval_iou'],
					bce_loss=metrics['bce_loss'], bb_loss=metrics['bb_loss'], train_bce_loss=metrics['train_bce_loss'],
					train_bb_loss=metrics['train_bb_loss'], train_accuracy=train_accuracy, train_iou=train_iou)

		if metrics['eval_accuracy'] > best_accuracy:
			best_accuracy = metrics['eval_accuracy']
			torch.save(mask_model.state_dict(), "./mask_model.pth")
		if metrics['eval_iou'] > best_iou:
			best_iou = metrics['eval_iou']
			torch.save(bb_model.state_dict(), "./bb_model.pth")

	return


@torch.no_grad()
def evaluate(bb_model: nn.Module, mask_model: nn.Module, dataloader: DataLoader) -> Scores:
	"""
	Evaluate a model without gradient calculation
	:param model: instance of a model
	:param dataloader: dataloader to evaluate the model on
	:return: tuple of (accuracy, loss) values
	"""
	accuracy_score = 0
	iou_score = 0
	loss = 0

	for i, (image, bounding_box, label, shape, path) in tqdm(enumerate(dataloader)):
		if torch.cuda.is_available():
			image = image.cuda()
			bounding_box = bounding_box.cuda()
			label = label.cuda()
			shape = shape.cuda()

		bb_hat = bb_model(image)
		label_hat = mask_model(image)
		# if np.random.random() > 0.95:
		# 	print_img_bb(bb_hat, bounding_box, shape, path, 'eval_sep_')
		bce_loss = nn.functional.binary_cross_entropy(label_hat.squeeze(-1), label)
		bb_loss = nn.functional.smooth_l1_loss(bb_hat, bounding_box)
		accuracy_score += accuracy(label_hat, label)
		iou_score += calc_iou(bb_hat, bounding_box, shape)

	bce_loss /= len(dataloader.dataset)
	bb_loss /= len(dataloader.dataset)
	accuracy_score /= len(dataloader.dataset)
	iou_score /= len(dataloader.dataset)
	accuracy_score *= 100
	iou_score *= 100

	return accuracy_score, iou_score, bce_loss.item(), bb_loss.item()


@torch.no_grad()
def predict(bb_model: nn.Module, mask_model: nn.Module, dataloader: DataLoader) -> Scores:
	"""
	predict a model without gradient calculation
	:param model: instance of a model
	:param dataloader: dataloader to evaluate the model on
	:return: predictions
	"""
	accuracy_score = 0
	iou_score = 0
	loss = 0
	bb_list = []
	pred_list = []
	files_list = []
	for i, (image, bounding_box, label, shape, path) in tqdm(enumerate(dataloader)):
		if torch.cuda.is_available():
			image = image.cuda()
			bounding_box = bounding_box.cuda()
			label = label.cuda()
			shape = shape.cuda()

		bb_hat = bb_model(image)
		label_hat = mask_model(image)

		# from relative bb to receiving format bb
		bb_hat = unscale_bb(bb_hat, shape)
		bb_hat = [[bb_hat[0][i].item(), bb_hat[1][i].item(), bb_hat[2][i].item(), bb_hat[3][i].item()] for i in
				  range(len(bb_hat[0]))]
		bb_list.append(bb_hat)

		pred_list.append([int(x) for x in (label_hat.squeeze() < 0.5)])
		files_list.append([x[x.rfind('/', 1) + 1:] for x in path])

	# flatten lists
	pred_list = [item for sublist in pred_list for item in sublist]
	files_list = [item for sublist in files_list for item in sublist]
	bb_list = [item for sublist in bb_list for item in sublist]
	return pred_list, files_list, bb_list
