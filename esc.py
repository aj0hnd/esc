import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Literal
from network import AllCNN, ResNet, ViT

class ESC(nn.Module):
    def __init__(
        self,
        p: float = 0.017,
        num_classes: int = 10,
        use_pretrain: bool = True,
        model_type: Literal['all_cnn', 'resnet', 'vit'] = 'all_cnn',
    ):
        super().__init__()
        
        self.p = p
        self.num_classes = num_classes
        self.use_pretrain = use_pretrain
        self.model_type = model_type
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_type == 'all_cnn':
            self.model_check_dir = './prepare/all_cnn_pretrained.pth' if use_pretrain else './prepare/all_cnn_retrained.pth'
            
            model = AllCNN(self.device, num_classes=num_classes)
            model.load_state_dict(torch.load(self.model_check_dir, map_location=self.device))
            
            self.feature_extractor = model.feature_extractor
            self.head = model.head
        
        elif model_type == 'resnet':
            self.model_check_dir = './prepare/resnet_pretrained.pth' if use_pretrain else './prepare/resnet_retrained.pth'
            
            model = ResNet(self.device, num_classes=num_classes)
            model.load_state_dict(torch.load(self.model_check_dir, map_location=self.device))
            
            self.feature_extractor = model.feature_extractor
            self.head = model.head
            
        elif model_type == 'vit':
            self.model_check_dir = './prepare/vit_pretrained.pth' if use_pretrain else './prepare/vit_retrained.pth'
            
            model = ViT(self.device)
            model.load_state_dict(torch.load(self.model_check_dir, map_location=self.device))
            
            self.feature_extractor = model.feature_extractor
            self.head = model.head
        
        self.last_feature_dim = model.last_feature_dim
        self.k = int(self.last_feature_dim * p)
        self.up = None
    
    def get_up_matrix(self, dataloader):
        all_activations = []
        for batch in tqdm(dataloader):
            batch_x, batch_y = batch
            with torch.no_grad():
                activation = self.feature_extractor(batch_x)
            
            all_activations += [activation]
        
        all_activations = torch.cat([*all_activations], dim=0)
        U, _, _ = torch.linalg.svd(all_activations.T, full_matrices=False)
        
        self.u = U
        self.up = U[:, self.k:]            
        print(f"\tMade Up matrix with p: {self.p}.\n")
    
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        
            if self.model_type == 'vit':
                x = x.pooler_output
            
            x = (self.up @ self.up.T @ x.T).T
            x = self.head(x)
            
        return x