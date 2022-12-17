import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def create_csv(number_of_train_folders = 34, number_of_frames=4):
    train_dataset = pd.DataFrame({'frames': [], 'label': []})
    for j in range(number_of_train_folders):
        lst = [
            f"Train{str(j+1).zfill(3)}/frame_{str(i+1).zfill(3)}.jpg" for i in range(200)]
        items = [(lst[i:i+number_of_frames], lst[i+number_of_frames]) for i in range(len(lst)-(number_of_frames+1))]
        x = pd.DataFrame(items, columns=["frames", "label"])
        train_dataset = train_dataset.append(x, ignore_index=True)
    return train_dataset

class AnomalyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (pd.DataFrame): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        current_sample = self.csv_file.iloc[idx, 0]
        label = self.csv_file.iloc[idx, 1]
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
    train_dataset = create_csv()
    anomaly_dataset = AnomalyDataset(train_dataset, root, None)
    data = DataLoader(anomaly_dataset, batch_size=16, shuffle=True)

    plt.rcParams["figure.figsize"] = [12, 12]
    plt.rcParams["figure.autolayout"] = True

    x, y, desc = next(iter(data))

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.title(f"frame_{i + 1}")
        plt.imshow(x[0, i, :, :], cmap="gray")
    print("example.png saved as output temporal frames")
    plt.savefig('example', dpi=200)