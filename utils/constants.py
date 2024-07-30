import torch

from utils.datasets import Toys4kDataset, CO3D, MVImgNet
from utils.losses import SimCLR, BYOL, VicReg
from torch.nn import functional as F

DATASETS = {
    "RT4K": {
            'class': Toys4kDataset,
            # 'size': 128664,
            'size': 375300,
            'img_size': (128, 128),
            'rgb_mean': (0.5, 0.5, 0.5),
            'rgb_std': (0.5, 0.5, 0.5),
            'action_size': 2
    },
    "MVImgNet": {
        'class': MVImgNet,
        'size': 6500000,
        'img_size': (224, 224),
        'rgb_mean': (0.485, 0.456, 0.406),
        'rgb_std': (0.229, 0.224, 0.225),
        'action_size': 8
    },
    "CO3D": {
        'class': CO3D,
        'size': 1500000,
        'img_size': (196, 196),
        'rgb_mean': (0, 0, 0),
        'rgb_std': (1, 1, 1),
        # 'action_size': 14
        'action_size': 9
    }
}


SIMILARITY_FUNCTIONS = {
    'cosine': lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2),
    'RBF': lambda x, x_pair: -torch.cdist(x, x_pair)
}

SIMILARITY_FUNCTIONS_SIMPLE = {
    'cosine': lambda x, x_pair: F.cosine_similarity(x, x_pair, dim=1),
    'RBF': lambda x, x_pair: -torch.norm(x - x_pair, 2)
}

# loss dictionary for different losses
LOSS = {
    'SimCLR': SimCLR,
    'BYOL': BYOL,
    'VicReg': VicReg
}
