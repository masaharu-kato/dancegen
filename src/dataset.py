# src/dataset.py

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

COND_DIM_FACE = 478 * 3
COND_DIM_POSE = 33 * 4


class CustomDanceDataset(Dataset):
    def __init__(self,
                    data_dir: Path | str,
                    # image_size,
                    transform=None,
                    *,
                    pose_only = False):
        self.data_dir = Path(data_dir)
        # self.image_size = image_size
        self.transform = transform
        self.pose_only = pose_only

        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.data_files = [f.replace(os.path.splitext(f)[1], '.npy') for f in self.image_files]

        if not self.image_files:
            raise ValueError(f"No image files found in {data_dir}. Please check your data directory.")

        if len(self.image_files) != len(self.data_files):
            print("Warning: Number of image files and data files do not match. Check your data preparation.")

        if self.transform is None:
            self.transform = transforms.Compose([
                # transforms.Resize(image_size),     # Already resized
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
            
    @property
    def cond_dim(self):
        return (COND_DIM_FACE if not self.pose_only else 0) + COND_DIM_POSE

    def __len__(self):
        return len(self.image_files) # 全てのファイル数を返す

    def __getitem__(self, idx): # idx は DataLoader から渡される生のインデックス
        img_name = self.image_files[idx] 
        data_name = self.data_files[idx] 

        img_path = self.data_dir / img_name
        data_path = self.data_dir / data_name

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image) # type: ignore

        conditional_data = np.load(data_path).astype(np.float32)
        if self.pose_only:
            conditional_data = conditional_data[COND_DIM_FACE:] # Extract only pose

        conditional_data = torch.from_numpy(conditional_data).flatten()

        return image, conditional_data
