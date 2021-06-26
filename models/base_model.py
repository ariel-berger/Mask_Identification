"""
    Example for a simple model
"""

from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
import torch


class MyModel(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, hidden_bb_dim = 100, hidden_label_dim=100, dropout: float = 0.2):
        super(MyModel, self).__init__()

        self.layers = []
        in_channels = 3  # RGB
        self.hidden_dims = [16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        # encoder image CNN layers:
        for idx, h_dim in enumerate(self.hidden_dims):
            # layers.append(
            steps =[nn.Conv2d(in_channels, out_channels=h_dim,
                    kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout2d(dropout)]
            if idx % 3 == 2:
                steps.append(nn.MaxPool2d(2))
            self.layers.append(nn.Sequential(*steps))
            in_channels = h_dim


        # self.conv_layers = nn.Sequential(*self.layers)
        self.fc_bb = nn.Sequential(
            nn.Linear(self.hidden_dims[-1]*7*7,hidden_bb_dim),
            nn.ReLU(),
            nn.Linear(hidden_bb_dim,4),
            nn.Sigmoid()
        )
        self.fc_label = nn.Sequential(
            nn.Linear(self.hidden_dims[-1]*7*7,hidden_label_dim),
            nn.ReLU(),
            nn.Linear(hidden_label_dim,1)
            ,nn.Sigmoid()
        )
        if torch.cuda.is_available():
            for layer in self.layers:
                layer.cuda()
        self.layers = nn.Sequential(*self.layers)

    def forward(self, image):
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        new = image
        for idx,layer in enumerate(self.layers):
            old = new
            new = layer(new)
            if idx % 3 ==  1: # residual connection
                new += old



        # conv = self.conv_layer(image)
        conv = new.view(image.size(0),-1)
        predicted_bb = self.fc_bb(conv)
        predicted_label = self.fc_label(conv)
        return predicted_bb, predicted_label
