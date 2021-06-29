"""
    Example for a simple model
"""

from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
import torch


class BBModel(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """

    def __init__(self, hidden_bb_dim=100, dropout: float = 0.2):
        super(BBModel, self).__init__()

        # stacking layers with pooling every 3 layers
        self.layers = []
        self.hidden_dims = [16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        in_channels = 3  # RGB
        for idx, h_dim in enumerate(self.hidden_dims):
            steps = [nn.Conv2d(in_channels, out_channels=h_dim,
                               kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(h_dim),
                     nn.LeakyReLU(),
                     nn.Dropout2d(dropout)]
            if idx % 3 == 2:
                steps.append(nn.MaxPool2d(2))
            self.layers.append(nn.Sequential(*steps))
            in_channels = h_dim

        # fully connected layer at the end of the CNN, to predict the bounding box
        self.fc_bb = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] * 7 * 7, hidden_bb_dim),
            nn.ReLU(),
            nn.Linear(hidden_bb_dim, 4),
            nn.Sigmoid()
        )

        if torch.cuda.is_available():
            for layer in self.layers:
                layer.cuda()
        self.layers = nn.Sequential(*self.layers)

    def forward(self, image):
        """
        :return: predicted image and label
        """
        # adding residual connections and forwarding the model through the CNN
        new = image
        for idx, layer in enumerate(self.layers):
            old = new
            new = layer(new)
            if idx % 3 == 1:  # residual connection
                new += old

        # Fully Connected layers to yield predictions
        conv = new.view(image.size(0), -1)
        predicted_bb = self.fc_bb(conv)
        return predicted_bb


class MaskModel(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """

    def __init__(self, hidden_label_dim=100, dropout: float = 0.2):
        super(MaskModel, self).__init__()

        # stacking layers with pooling every 3 layers
        self.layers = []
        self.hidden_dims = [16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        in_channels = 3  # RGB
        for idx, h_dim in enumerate(self.hidden_dims):
            steps = [nn.Conv2d(in_channels, out_channels=h_dim,
                               kernel_size=3, stride=1, padding=1),
                     nn.BatchNorm2d(h_dim),
                     nn.LeakyReLU(),
                     nn.Dropout2d(dropout)]
            if idx % 3 == 2:
                steps.append(nn.MaxPool2d(2))
            self.layers.append(nn.Sequential(*steps))
            in_channels = h_dim

        # fully connected layer at the end of the CNN, to predict the label

        self.fc_label = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] * 7 * 7, hidden_label_dim),
            nn.ReLU(),
            nn.Linear(hidden_label_dim, 1)
            , nn.Sigmoid()
        )

        if torch.cuda.is_available():
            for layer in self.layers:
                layer.cuda()
        self.layers = nn.Sequential(*self.layers)

    def forward(self, image):
        """
        :return: predicted image and label
        """
        # adding residual connections and forwarding the model through the CNN
        new = image
        for idx, layer in enumerate(self.layers):
            old = new
            new = layer(new)
            if idx % 3 == 1:  # residual connection
                new += old

        # Fully Connected layers to yield predictions
        conv = new.view(image.size(0), -1)
        predicted_label = self.fc_label(conv)
        return predicted_label
