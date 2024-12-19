import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
from torchvision import transforms

class RankIQADataset(Dataset):
    def __init__(self, json_file, transform=None):
        """
        JSON 文件格式：{"img1_path": "path_to_image1", "img2_path": "path_to_image2", "label": 1 or -1}
        """
        with open(json_file, 'r') as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading JSON file: {e}")
                raise e
        self.transform = transform
        print("dataset loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1_path = self.data[idx]['img1_path']
        img2_path = self.data[idx]['img2_path']
        label = self.data[idx]['label']

        # 检查文件路径是否有效
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            raise FileNotFoundError(f"Image file not found: {img1_path} or {img2_path}")

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
