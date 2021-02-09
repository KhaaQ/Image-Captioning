# Imports
import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader # Gives easier dataset managment and creates mini batches

class ValData(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # def __len__(self):
    #     return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir)
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

# Load Data
dataset = ValData(root_dir = 'cats_dogs_resized',
                             transform = transforms.ToTensor())

# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.


test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
