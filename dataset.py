"""
Here, we create a custom dataset
"""
import torch

from utils.types import PathT
from torch.utils.data import Dataset
from typing import Any, Tuple
from PIL import Image
import glob
import ast
from torchvision import transforms


class MaskDataset(Dataset):
    """
    Mask dataset. contains the images, the bounding box and the label
    """

    def __init__(self, path: PathT) -> None:
        # Set variables
        self.folder_path = path

        # Load features
        self.image_list = self._get_features()

    def __getitem__(self, index: int) -> Tuple:
        """
        :return: image, bounding box, label, shape
        """
        return self.image_list[index][0], self.image_list[index][1], self.image_list[index][2], self.image_list[index][
            3], self.image_list[index][4]

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.image_list)

    def _get_features(self) -> Any:
        """
        :return: list of lists with the features of the images,
        [[image1, bounding box1, label1, shape1],[image2, bounding box2, label2, shape2],...] as values
        """

        image_list = []
        for filename in glob.glob(self.folder_path + '/*.jpg'):
            im = Image.open(filename)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize, ])

            # extracting bb, label and shape
            closer_location = filename.find(']') + 1
            opener_location = filename.find('[')
            bounding_box = torch.tensor(ast.literal_eval(filename[opener_location:closer_location]),
                                        requires_grad=False, dtype=torch.int32)
            shape = torch.tensor(im.size, requires_grad=False, dtype=torch.int32)
            label = torch.tensor(ast.literal_eval(filename[closer_location + 2:-4]), requires_grad=False,
                                 dtype=torch.float32)
            image = transform(im.copy())

            # scaling the shape of the bounding box to [0,1] scale
            relative_bb = self.scale_bb(bounding_box, shape)
            image_list.append([image, relative_bb, label, shape, filename])
            im.close()
        return image_list

    @staticmethod
    def scale_bb(bounding_box, shape):
        """

        :param bounding_box: location of the bounding box
        :param shape: original shape of the image
        :return: scales version of the bounding box
        """
        rel_x = bounding_box[0] / shape[0]
        rel_y = bounding_box[1] / shape[1]
        rel_width = bounding_box[2] / shape[0]
        rel_height = bounding_box[3] / shape[1]
        return torch.tensor([rel_x, rel_y, rel_width, rel_height], requires_grad=False, dtype=torch.float32)