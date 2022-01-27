import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ToyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 5)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]