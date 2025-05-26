import os
import torch
import numpy as np

from PIL import Image
from glob import glob
from typing import Literal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10, CIFAR100

class CustomDataset(Dataset):
    def __init__(self, x, y, transform):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.transform(self.x[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return (x, y)
    
class CustomDatasetPath(Dataset):
    def __init__(self, x, y, class_dict, transform):
        super().__init__()
        self.x = x
        self.y = y
        self.class_dict = class_dict
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.transform(Image.open(self.x[idx]).convert('RGB'))
        # x = Image.open(self.x[idx]).convert('RGB')
        y = torch.tensor(self.class_dict[self.y[idx]], dtype=torch.long)
        return (x, y)
        

class InputPipeLineBuilder:
    dataset_dir = {
        'cifar10': './data/cifar10',
        'cifar100': './data/cifar100',
        'tiny_imagenet': './data/tiny_imagenet'
    }
    
    def __init__(
        self, batch_size: int, valid_size: int=0.2, 
        select_forget_concept: bool = False,
        dataset: Literal['cifar10', 'cifar100', 'tiny_imagenet'] = 'cifar10'
        ):
        
        self.batch_size = batch_size
        self.valid_size = 0.9 if select_forget_concept else valid_size
        self.dataset_type = dataset
        self.select_forget_concept = select_forget_concept
        root_dir = InputPipeLineBuilder.dataset_dir[dataset]
        
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ])
        test_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ])
        
        if dataset == 'cifar10':
            
            train_ds = CIFAR10(root=root_dir, train=True, download=True)
            test_ds = CIFAR10(root=root_dir, train=False, download=True)
            
            self.classes = train_ds.classes
            self.classes_to_idx = train_ds.class_to_idx
            
            if select_forget_concept == False:
                
                train_x, valid_x, train_y, valid_y = train_test_split(train_ds.data, train_ds.targets, test_size=valid_size, shuffle=True)
                
                self.ds = {
                    'train': CustomDataset(train_x, train_y, train_transform),
                    'valid': CustomDataset(valid_x, valid_y, test_transform),
                    'test': CustomDataset(test_ds.data, test_ds.targets, test_transform)
                }
                
            else:
                # 편의상 그냥 0번째 클래스로 고정했음
                self.forget_classes = [self.classes[0]]
                self.retain_classes = self.classes[1:]
                
                self.forget_classes_to_idx = {self.forget_classes[0]: 0}
                self.retain_classes_to_idx = {cls: i+1 for i, cls in enumerate(self.classes[1:])}
                
                mask = (np.array(train_ds.targets) == 0)
                
                forget_train_x = train_ds.data[mask]
                forget_train_y = np.array(train_ds.targets)[mask]
                forget_train_x, forget_valid_x, forget_train_y, forget_valid_y = train_test_split(forget_train_x, forget_train_y, 
                                                                                                  test_size=self.valid_size, shuffle=True)
                
                retain_train_x = train_ds.data[~mask]
                retain_train_y = np.array(train_ds.targets)[~mask]
                retain_train_x, retain_valid_x, retain_train_y, retain_valid_y = train_test_split(retain_train_x, retain_train_y, 
                                                                                                  test_size=0.2, shuffle=True)
                
                mask = (np.array(test_ds.targets) == 0)
                
                forget_test_x = test_ds.data[mask]
                forget_test_y = np.array(test_ds.targets)[mask]
                
                retain_test_x = test_ds.data[~mask]
                retain_test_y = np.array(test_ds.targets)[~mask]
                
                self.ds = [ # retain, forget 순
                    {
                        'train': CustomDataset(retain_train_x, retain_train_y, train_transform),
                        'valid': CustomDataset(retain_valid_x, retain_valid_y, test_transform),
                        'test': CustomDataset(retain_test_x, retain_test_y, test_transform) 
                    },
                    {
                        'train': CustomDataset(forget_train_x, forget_train_y, train_transform),
                        'valid': CustomDataset(forget_valid_x, forget_valid_y, test_transform),
                        'test': CustomDataset(forget_test_x, forget_test_y, test_transform)
                    }
                ]
                
            
        elif dataset == 'cifar100':
            
            train_ds = CIFAR100(root=root_dir, train=True, download=False)
            test_ds = CIFAR100(root=root_dir, train=False, download=False)
            
            self.classes = train_ds.classes
            self.classes_to_idx = train_ds.class_to_idx
            
            if select_forget_concept == False:
                
                train_x, valid_x, train_y, valid_y = train_test_split(train_ds.data, train_ds.targets, 
                                                                      test_size=self.valid_size, shuffle=True)
                
                self.ds = {
                    'train': CustomDataset(train_x, train_y, train_transform),
                    'valid': CustomDataset(valid_x, valid_y, test_transform),
                    'test': CustomDataset(test_ds.data, test_ds.targets, test_transform)
                }
                
            else:
                self.forget_classes = [self.classes[i*10] for i in range(10)]
                self.retain_classes = [self.classes[i*10+j] for i in range(10) for j in range(1, 10)]
                
                self.forget_classes_to_idx = {self.forget_classes[i] : i*10 for i in range(10)}
                self.retain_classes_to_idx = {self.classes[i*10+j] : i*10+j for i in range(10) for j in range(1, 10)}
                
                mask = (np.array(train_ds.targets) % 10 == 0)
                
                forget_train_x = train_ds.data[mask]
                forget_train_y = np.array(train_ds.targets)[mask]
                forget_train_x, forget_valid_x, forget_train_y, forget_valid_y = train_test_split(forget_train_x, forget_train_y, 
                                                                                                  test_size=self.valid_size, shuffle=True)
                retain_train_x = train_ds.data[~mask]
                retain_train_y = np.array(train_ds.targets)[~mask]
                retain_train_x, retain_valid_x, retain_train_y, retain_valid_y = train_test_split(retain_train_x, retain_train_y,
                                                                                                  test_size=0.2, shuffle=True)
                mask = (np.array(test_ds.targets) % 10 == 0)
                
                forget_test_x = test_ds.data[mask]
                forget_test_y = np.array(test_ds.targets)[mask]
                
                retain_test_x = test_ds.data[~mask]
                retain_test_y = np.array(test_ds.targets)[~mask]
                
                self.ds = [ # retain, forget 순
                    {
                        'train': CustomDataset(retain_train_x, retain_train_y, train_transform),
                        'valid': CustomDataset(retain_valid_x, retain_valid_y, test_transform),
                        'test': CustomDataset(retain_test_x, retain_test_y, test_transform) 
                    },
                    {
                        'train': CustomDataset(forget_train_x, forget_train_y, train_transform),
                        'valid': CustomDataset(forget_valid_x, forget_valid_y, test_transform),
                        'test': CustomDataset(forget_test_x, forget_test_y, test_transform)
                    }
                ]
            
        elif dataset == 'tiny_imagenet':
            
            train_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(dtype=torch.float32, scale=True)
            ])
            test_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(dtype=torch.float32, scale=True)
            ])
            
            with open(root_dir+'/wnids.txt', 'r', encoding='utf-8') as f:
                self.classes = sorted([line.strip() for line in f])
            self.classes_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            
            if select_forget_concept == False:
                train_x, train_y = [], []
                for class_name in self.classes:
                    class_paths = glob(os.path.join(root_dir, 'train', class_name, 'images', '*.JPEG'))
                    train_x += class_paths
                    train_y += [class_name] * len(class_paths)
            
                train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_size, shuffle=True)
                
                test_x = sorted(glob(os.path.join(root_dir, 'val/images/*.JPEG')))
                test_y = []
                with open(root_dir+'/val/val_annotations.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        test_y += [line.split()[1]]
            
                self.ds = {
                    'train': CustomDatasetPath(train_x, train_y, self.classes_to_idx, train_transform),
                    'valid': CustomDatasetPath(valid_x, valid_y, self.classes_to_idx, test_transform),
                    'test': CustomDatasetPath(test_x, test_y, self.classes_to_idx, test_transform)
                }
            
            else:
                self.forget_classes = [self.classes[20*i] for i in range(10)]
                self.retain_classes = [self.classes[20*i+j] for i in range(10) for j in range(1, 20)]
                
                self.forget_classes_to_idx = {self.forget_classes[i] : i*20 for i in range(20)}
                self.retain_classes_to_idx = {self.classes[i*20+j] : i*20+j for i in range(20) for j in range(1, 20)}
                
                forget_train_x, forget_train_y, retain_train_x, retain_train_y = [], [], [], []
                for i, class_name in enumerate(self.classes):
                    class_paths = glob(os.path.join(root_dir, 'train', class_name, 'images', '*.JPEG'))
                    if i % 20 == 0:
                        forget_train_x += class_paths
                        forget_train_y += [class_name] * len(class_paths)
                    else:
                        retain_train_x += class_paths
                        retain_train_y += [class_name] * len(class_paths)
                
                forget_train_x, forget_valid_x, forget_train_y, forget_valid_y = train_test_split(forget_train_x, forget_train_y, 
                                                                                                  test_size=self.valid_size, shuffle=True)
                retain_train_x, retain_valid_x, retain_train_y, retain_valid_y = train_test_split(retain_train_x, retain_train_y, 
                                                                                                  test_size=0.2, shuffle=True)
                
                test_x = sorted(glob(os.path.join(root_dir, 'val/images/*.JPEG')))
                test_y = []
                with open(root_dir+'/val/val_annotations.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        test_y += [line.split()[1]]
                forget_test_x, forget_test_x, retain_test_x, retain_test_y = [], [], [], []
                for i, test_class in enumerate(test_y):
                    if test_class in self.forget_classes:
                        forget_test_x += [test_x[i]]
                        forget_test_y += [test_class]
                    else:
                        retain_test_x += [test_x[i]]
                        retain_test_y += [test_class]
                
                self.ds = [ # retain, forget 순
                    {
                        'train': CustomDatasetPath(retain_train_x, retain_train_y, self.retain_classes_to_idx, train_transform),
                        'valid': CustomDatasetPath(retain_valid_x, retain_valid_y, self.retain_classes_to_idx, self.test_transform),
                        'test': CustomDatasetPath(retain_test_x, retain_test_y, self.retain_classes_to_idx, test_transform) 
                    },
                    {
                        'train': CustomDataset(forget_train_x, forget_train_y, self.forget_classes_to_idx, train_transform),
                        'valid': CustomDataset(forget_valid_x, forget_valid_y, self.forget_classes_to_idx, test_transform),
                        'test': CustomDataset(forget_test_x, forget_test_y, self.forget_classes_to_idx, test_transform)
                    }
                ]
        
    def get_dataloader(self, subset='train'):
        assert subset in ['train', 'valid', 'test'], f"subset support: ['train', 'valid', 'test']"
        
        shuffle = True if subset == 'train' else False
        drop_last = True if subset == 'train' else False
        return DataLoader(self.ds[subset], batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)
    
    def get_dataloader_for_unlearn(self, is_retain=True, subset='train'):
        assert subset in ['train', 'valid', 'test'], f"subset support: ['train', 'valid', 'test']"
        
        shuffle = True if subset == 'train' else False
        drop_last = True if subset == 'train' else False
        idx = 0 if is_retain else 1
        return DataLoader(self.ds[idx][subset], batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)