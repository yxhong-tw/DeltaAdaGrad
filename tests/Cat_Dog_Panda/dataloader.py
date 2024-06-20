import os

import PIL.Image
import cv2
import torch
import matplotlib.pyplot as plt
import random
import shutil
import PIL

from torchvision import datasets,models,transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split

def load_data(filepath, batch_size):

    def shuffled_filename(filepath):
        cats = os.listdir(filepath + '/cats')
        dogs = os.listdir(filepath + '/dogs')
        panda = os.listdir(filepath + '/panda')

        if not os.path.exists("tests/Cat_Dog_Panda/input"):
            os.makedirs("tests/Cat_Dog_Panda/input")

        # copy all images from animals to input folder
        for i in range(len(cats)):
            shutil.copy(filepath + '/cats/' + cats[i], 'tests/Cat_Dog_Panda/input')
            shutil.copy(filepath + '/dogs/' + dogs[i], 'tests/Cat_Dog_Panda/input')
            shutil.copy(filepath + '/panda/' + panda[i], 'tests/Cat_Dog_Panda/input')
        
        filename_list = cats + dogs + panda
        for i in range(len(filename_list)):
            filename_list[i] = 'tests/Cat_Dog_Panda/input' + '/' + filename_list[i]
        random.shuffle(filename_list)

        return filename_list


    transform1 = transforms.Compose(
        [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor()
        ]
    )    

    transform2= transforms.Compose(
        [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ]
    )
    
    transform3 = transforms.Compose(
        [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor()
        ]   
    )

    transform_val_test = transforms.Compose(
        [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ]
    )


    class dataset(Dataset):
        def __init__(self, filelist, transform=None):
            self.filelist = filelist
            self.transform = transform

        def __len__(self):
            self.file_len = len(self.filelist)
            return self.file_len
        
        def __getitem__(self, idx):
            img_name = self.filelist[idx]
            image = PIL.Image.open(img_name).convert('RGB')
            image = self.transform(image)

            if 'cat' in img_name:
                label = 0
            elif 'dog' in img_name:
                label = 1
            else:
                label = 2
            
            return image, label
        
    train_list, val_list = train_test_split(shuffled_filename(filepath), test_size=0.2)

    dataset_train_1 = dataset(train_list, transform1)
    dataset_train_2 = dataset(train_list, transform2)
    dataset_train_3 = dataset(train_list, transform3)

    dataset_train = torch.utils.data.ConcatDataset([dataset_train_1, dataset_train_2, dataset_train_3]) 

    dataset_val = dataset(val_list, transform_val_test)

    train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size, shuffle=True)

    return train_loader, val_loader
