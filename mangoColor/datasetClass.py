import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class MangoColorDataset(Dataset):
    def __init__(self, image_path: str, model=None):
        self.image_path = image_path
        self.image_filenames = os.listdir(image_path)
        self.model = model
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        file = os.path.join(self.image_path, self.image_filenames[idx])

        if self.model is None:
            raise Exception("model needs to be a MangoColor class object and must be supplied")
        
        input_image, real_image = self.model.load(file)
        input_image, real_image = self.model.random_jitter(input_image, real_image)

        return input_image, real_image




