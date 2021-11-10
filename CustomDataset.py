# BUILD CUSTOM DATASETS


# PyTorch gives you the freedom to pretty much do anything with the Dataset class so long as you override two of the subclass functions:

# the len function which returns the size of the dataset, and
# the getitem function which returns a sample from the dataset given an index


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader , Dataset , random_split
from torchvision import transforms
import os
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt


root_dir = './datasets/catsanddogs/'
csv_file = './datasets/cats_dogs.csv'

pd.read_csv(csv_file).head()


class CatsandDogDataset(Dataset):   
    def __init__(self , root_dir , csv_file , transforms = None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotations = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self , index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index , 0])
        img = io.imread(img_path)
        label = torch.tensor(int(self.annotations.iloc[index , 1]))
        if self.transforms:
            image = self.transforms(img)
            
        return (image , label)
    
    
dataset = CatsandDogDataset(root_dir=root_dir, 
                            csv_file=csv_file, 
                            transforms=transforms.ToTensor())


train_set , test_set = random_split(dataset , [8 , 2])


train_loader = DataLoader(train_set , batch_size=1 , shuffle=True)
test_loader = DataLoader(test_set , batch_size=1 , shuffle=False)


print(len(train_loader) , len(test_loader))


images , labels = next(iter(train_loader))
print(images.shape , labels.shape)


io.imshow(torch.permute(images[0] , (1, 2, 0)).numpy())
plt.axis(False)
plt.show()