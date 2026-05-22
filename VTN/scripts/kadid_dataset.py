# E:\MRVNet2D\Restormer + Volterra\kadid_dataset.py
# kadid_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

class KADID10KDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.distorted_dir = 'E:/MRVNet2D/dataset/KADID10K/images'
        self.reference_dir = 'E:/MRVNet2D/dataset/KADID10K/images'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        distorted_name = self.data.iloc[idx, 0]
        reference_name = self.data.iloc[idx, 1]

        distorted_path = os.path.join(self.distorted_dir, distorted_name)
        reference_path = os.path.join(self.reference_dir, reference_name)

        distorted = Image.open(distorted_path).convert("RGB")
        reference = Image.open(reference_path).convert("RGB")

        if self.transform:
            distorted = self.transform(distorted)
            reference = self.transform(reference)

        return distorted, reference
