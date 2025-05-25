import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTModel

class AllCNN(nn.Module):
    def __init__(self, last_feature_dim=10, num_classes=10):
       super().__init__()
       
       self.last_feature_dim = last_feature_dim
       self.feature_extractor = nn.Sequential(
           nn.Conv2d(3, 96, 3, 1, 1, bias=False),
           nn.BatchNorm2d(96),
           nn.ReLU(inplace=True),
           
           nn.Conv2d(96, 96, 3, 1, 1, bias=False),
           nn.BatchNorm2d(96),
           nn.ReLU(inplace=True),
           
           nn.MaxPool2d(3, 2, 1),
           
           nn.Conv2d(96, 192, 3, 1, 1, bias=False),
           nn.BatchNorm2d(192),
           nn.ReLU(inplace=True),
           
           nn.Conv2d(192, 192, 1, 1, 0, bias=False),
           nn.BatchNorm2d(192),
           nn.ReLU(inplace=True),
           
           nn.Conv2d(192, 10, 1, 1, 0, bias=False),
           nn.BatchNorm2d(10),
           nn.ReLU(inplace=True),
           
           nn.AdaptiveAvgPool2d((1, 1)),
           nn.Flatten(start_dim=1)
       )
       
       self.head = nn.Linear(last_feature_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        
        return x

class ResNet(nn.Module):
    def __init__(self, last_feature_dim=512, num_classes=100):
        super().__init__()
        
        self.last_feature_dim = last_feature_dim
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(
            *list(base.children())[:-1],
            nn.Flatten(start_dim=1)
        )
        self.head = nn.Linear(last_feature_dim, num_classes)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        
        return x
    
class ViT(nn.Module):
    def __init__(self, last_feature_dim=768, checkpoint='google/vit-base-patch16-224-in21k', num_clases=200):
        super().__init__()
        
        self.last_feature_dim = last_feature_dim
        self.feature_extractor = ViTModel.from_pretrained(checkpoint)
        self.head = nn.Linear(last_feature_dim, num_clases)
    def forward(self, x):
        x = self.feature_extractor(x).pooler_output
        x = self.head(x)
        
        return x