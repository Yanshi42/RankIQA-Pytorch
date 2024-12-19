import torch
from torchvision import transforms
from PIL import Image

class Preprocessor:
    def __init__(self, patch_num=30):
        self.patch_num = patch_num
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return self.transform(x)
