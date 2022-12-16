import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

Train_DataSet = pd.DataFrame({'frames': [], 'label': []})
for j in range(34):
  lst = [f"Train{str(j+1).zfill(3)}/frame_{str(i+1).zfill(3)}.jpg" for i in range(200)]
  items = [(lst[i:i+4], lst[i+4]) for i in range(len(lst)-5)]
  x = pd.DataFrame(items, columns=["frames", "label"])
  Train_DataSet = Train_DataSet.append(x, ignore_index=True)

class AnomalyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.Train_DataSet = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Train_DataSet)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        current_sample = self.Train_DataSet.iloc[idx, 0]
        label = Train_DataSet.iloc[idx, 1]
        label_path = os.path.join(self.root_dir, label)
        label = io.imread(label_path)
        images = []
        for item in current_sample:
          img_name = os.path.join(self.root_dir, item)
          image = io.imread(img_name)
          image = transform.resize(image, (256, 256))
          images.append(image)

        label = transform.resize(label, (256, 256))
        x = np.stack(images[:4])

        return x, label, images

if __name__ == "__main__":
    root = "/TrainOpticalFlow"
    data_transform = transforms.Compose([transforms.ToTensor()])
    anomaly_dataset = AnomalyDataset(Train_DataSet, root, None)
    loader = DataLoader(anomaly_dataset, batch_size=16, shuffle=True)