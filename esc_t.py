import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Literal
from network import AllCNN, ResNet, ViT
from esc import ESC

class ESC_T(ESC):
    def __init__(
        self, 
        threshold: float,
        p: float = 0.017,
        num_classes: int = 10,
        use_pretrain: bool = True,
        model_type: Literal['all_cnn', 'resnet', 'vit'] = 'all_cnn',
    ):
        super().__init__(p, num_classes, use_pretrain, model_type)
        
        self.threshold = threshold
        self.mask = nn.Parameter(torch.ones(size=(self.last_feature_dim, self.last_feature_dim), dtype=torch.float32))
        
        for layer in self.modules():
            layer.requires_grad_ = False
            
        self.mask.requires_grad_ = True
        
    def threshold_mask(self):
        condition = (self.mask > self.threshold).float()
        self.mask.data = condition
    
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
            
            if self.model_type == 'vit':
                x = x.pooler_output
        
        ur = self.u * self.mask
        x = (ur @ ur.T @ x.T).T
        
        x = self.head(x)
        return x
    
    def train_mask(self, dataloader, optimizer, num_epochs, output_dir='./prepare/esct_mask'):
        total_losses = []
        
        self.train()
        for epoch in range(num_epochs):
            losses = []
            for batch in tqdm(dataloader):
                batch_x, batch_y = batch
                logits = self.forward(batch_x.to(self.device))
                
                preds = torch.argmax(logits, dim=1)
                wrong_mask = (preds == batch_y.to(self.device))
                
                loss = -F.cross_entropy(logits, batch_y)
                loss = loss * wrong_mask
                
                if wrong_mask.sum() == 0:
                    loss = 0.0
                    losses.append(loss)

                else:
                    selected_logits = logits[wrong_mask]
                    selected_labels = batch_y[wrong_mask]
                    loss = -F.cross_entropy(selected_logits, selected_labels)
                    
                    loss.backward()
                    losses.append(loss.cpu().item())
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                with torch.no_grad():
                    self.mask.clamp_(0, 1)
                

            total_losses += [losses]
            print(f"\tavg loss at epoch: {epoch+1}/{num_epochs} : {sum(losses)/len(losses):.4f}\n")
        
        self.threshold_mask()
        torch.save(self.mask.data.clone(), f"{output_dir}_{self.model_type}.pth")
            
        return total_losses