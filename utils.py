import os
import math
import torch
import torch.nn as nn

from tqdm import tqdm
from network import AllCNN

def get_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    for batch in dataloader:
        batch_x, batch_y = batch
        with torch.no_grad():
            logits = model(batch_x.to(model.device))
            preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y.to(model.device)).sum().item()
        total += batch_x.shape[0]
    
    return correct / total

def get_hm(remain_acc, forget_acc):
    forget_suc = 1 - forget_acc
    if remain_acc + forget_suc == 0:
        return 0.0
    return 2 * (remain_acc * forget_suc) / (remain_acc + forget_suc)

def get_mia(train_dataloader, test_dataloader, model):
    train_prob, test_prob = [], []
    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    for batch in train_dataloader:
        batch_x, batch_y = batch
        with torch.no_grad():
            logits = model(batch_x.to(model.device))
            preds = softmax(logits)
            true_preds = torch.gather(preds, 1, batch_y.view(-1, 1).to(model.device)).squeeze(1)
            train_prob.extend(true_preds.cpu().tolist())
            
    for batch in test_dataloader:
        batch_x, batch_y = batch
        with torch.no_grad():
            logits = model(batch_x.to(model.device))
            preds = softmax(logits)
            true_preds = torch.gather(preds, 1, batch_y.view(-1, 1).to(model.device)).squeeze(1)
            test_prob.extend(true_preds.cpu().tolist())
    
    count = 0
    for train_pred in train_prob:
        for test_pred in test_prob:
            if train_pred > test_pred:
                count += 1.0
            elif math.isclose(train_pred, test_pred, rel_tol=1e-5):
                count += 0.5
    
    total = len(train_prob) * len(test_prob)
    mia_auc = count / total
    return mia_auc
    
def evaluate(r_dataloader, f_dataloader, rt_dataloader, ft_dataloader, model):
    
    acc_r = get_accuracy(model, r_dataloader)
    acc_f = get_accuracy(model, f_dataloader)
    acc_rt = get_accuracy(model, rt_dataloader)
    acc_ft = get_accuracy(model, ft_dataloader)
    
    hm_train = get_hm(acc_r, acc_f)
    hm_test = get_hm(acc_rt, acc_ft)
    
    mia = get_mia(f_dataloader, ft_dataloader, model)
    
    return {
        'd_f': round(acc_f, 2),
        'd_r': round(acc_r, 2),
        'd_ft': round(acc_ft, 2),
        'd_rt': round(acc_rt, 2),
        'hm': round(hm_train, 2),
        'hm_t': round(hm_test, 2),
        'mia': round(mia, 2)
    }

def evaluate_kr(r_dataloader, f_dataloader, rt_dataloader, ft_dataloader, model, output_dir, use_esc=False):
    for param in model.feature_extractor.parameters():
        param.requires_grad_ = False
    feature_dim = model.last_feature_dim
    num_classes = model.num_classes
    
    linear_probe = nn.Linear(feature_dim, num_classes).to(model.device)
    
    if os.path.exists(output_dir):
        linear_probe.load_state_dict(torch.load(output_dir, map_location=model.device))
    
    else:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(linear_probe.parameters(), lr=0.001)
        
        
        for epoch in range(10):
            for batch in tqdm(f_dataloader):
                batch_x, batch_y = batch
                with torch.no_grad():
                    features = model.feature_extractor(batch_x.to(model.device))
                logits = linear_probe(features)
                loss = loss_fn(logits, batch_y.to(model.device))
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            for batch in tqdm(r_dataloader):
                    batch_x, batch_y = batch
                    with torch.no_grad():
                        features = model.feature_extractor(batch_x.to(model.device))
                    logits = linear_probe(features)
                    loss = loss_fn(logits, batch_y.to(model.device))
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        torch.save(linear_probe.state_dict(), output_dir)
        
    temp_model = nn.Sequential(model.feature_extractor, linear_probe)
    
    acc_f = get_accuracy(temp_model, f_dataloader)
    acc_r = get_accuracy(temp_model, r_dataloader)
    acc_ft = get_accuracy(temp_model, ft_dataloader)
    acc_rt = get_accuracy(temp_model, rt_dataloader)
    
    hm_train = get_hm(acc_r, acc_f)
    hm_test = get_hm(acc_rt, acc_ft)
    
    return {
        'd_f': round(acc_f, 2),
        'd_r': round(acc_r, 2),
        'd_ft': round(acc_ft, 2),
        'd_rt': round(acc_rt, 2),
        'hm': round(hm_train, 2),
        'hm_t': round(hm_test, 2),
    }