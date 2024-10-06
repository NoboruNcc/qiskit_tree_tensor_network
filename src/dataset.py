import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SimpleToTensor:
    def __init__(self, dtype=int):
        self.dtype=dtype

    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def transform_label(label):
    return 1 if label == 1 else -1

target_transform = transforms.Compose([
    SimpleToTensor(),
    transform_label
])

class TransformableDataset(Dataset):
    def __init__(self,data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        
        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y
