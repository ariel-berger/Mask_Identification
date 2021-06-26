"""
Here, we create a custom dataset
"""
import torch
import pickle

from utils.types import PathT
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List
from PIL import Image
import glob
import ast
from torchvision import transforms

class MaskDataset(Dataset):
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, path: PathT) -> None:
        # Set variables
        self.folder_path = path

        # Load features
        self.image_list = self._get_features()
        # self.features = self._get_features()

        # Create list of entries
        # self.entries = self._get_entries()

    def __getitem__(self, index: int) -> Tuple:
        """

        :param index:
        :return: image, bounding box, label
        """
        return self.image_list[index][0], self.image_list[index][1],self.image_list[index][2],self.image_list[index][3]

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.image_list)

    def _get_features(self) -> Any:
        """
        :return: dict with image name as key and [image, bounding box, label] as values
        """

        image_list =[]
        for filename in glob.glob(self.folder_path +'/*.jpg'):
            im = Image.open(filename)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize, ])

            closer_location = filename.find(']') + 1
            opener_location = filename.find('[')
            bounding_box = torch.tensor(ast.literal_eval(filename[opener_location:closer_location]), requires_grad=False,dtype=torch.int32)
            shape = torch.tensor(im.size, requires_grad=False,dtype=torch.int32)
            label = torch.tensor(ast.literal_eval(filename[closer_location + 2:-4]), requires_grad=False, dtype=torch.float32)
            image = transform(im.copy())
            relative_bb = self.scale_bb(bounding_box,shape)
            image_list.append([image,relative_bb,label, shape])
            im.close()
        return image_list

    @staticmethod
    def scale_bb(bounding_box,shape ):
        rel_x = bounding_box[0] / shape[0]
        rel_y = bounding_box[1] / shape[1]
        rel_width = bounding_box[2] / shape[0]
        rel_height = bounding_box[3] / shape[1]
        return torch.tensor([rel_x,rel_y, rel_width, rel_height], requires_grad=False,dtype=torch.float32)
    # def _get_entries(self) -> List:
    #     """
    #     This function create a list of all the entries. We will use it later in __getitem__
    #     :return: list of samples
    #     """
    #     entries = []
    #
    #     for idx, item in self.features.items():
    #         entries.append(self._get_entry(item))
    #
    #     return entries
    #
    # @staticmethod
    # def _get_entry(item: Dict) -> Dict:
    #     """
    #     :item: item from the data. In this example, {'input': Tensor, 'y': int}
    #     """
    #     x = item['input']
    #     y = torch.Tensor([1, 0]) if item['label'] else torch.Tensor([0, 1])
    #
    #     return {'x': x, 'y': y}
