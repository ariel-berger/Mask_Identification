import os
import argparse
import numpy as np
import pandas as pd
from models.base_model import ResidualModel
from models.separate_model import MaskModel, BBModel
# from train import predict, evaluate
from train_separately import predict, evaluate
import torch
from dataset import MaskDataset
from torch.utils.data import DataLoader

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)
# load model
config = {}
config["batch_size"] = 16  # 32
config["dropout"] = 0.098203  # 0.13303
config["lr_value"] = 0.00034058  # 0.000078933
config["hidden_bb_dim"] = 298  # 340
config["hidden_label_dim"] = 259  # 239
config["bb_loss_weight"] = 28  # 86
config["step_size"] = 30

# model = ResidualModel(dropout=config['dropout'], hidden_bb_dim=config['hidden_bb_dim'],
#                       hidden_label_dim=config['hidden_label_dim'])
mask_model = MaskModel(dropout=config['dropout'], hidden_label_dim=config['hidden_label_dim'])
bb_model = BBModel(dropout=config['dropout'], hidden_bb_dim=config['hidden_bb_dim'])

# ##load model
# model.load_state_dict(torch.load('model.pth'))
mask_model.load_state_dict(torch.load('mask_model.pth'))
bb_model.load_state_dict(torch.load('bb_model.pth'))

if torch.cuda.is_available():
	# model = model.cuda()
	mask_model = mask_model.cuda()
	bb_model = bb_model.cuda()

# Load dataset
test_dataset = MaskDataset(args.input_folder)

test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=True,
						 num_workers=6)

# accuracy, iou, loss, _ = evaluate(bb_model, mask_model, test_loader)
# print(f"Accuracy score is {accuracy}, IOU score is {iou}, loss is {loss}")

pred_list, files_list, bb_list = predict(bb_model, mask_model, test_loader)

prediction_df = pd.DataFrame(zip(files_list, *list(map(list, zip(*bb_list))), pred_list),
							 columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
####

prediction_df.to_csv("prediction.csv", index=False, header=True)
