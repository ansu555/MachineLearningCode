import torch
import numpy as np
import os
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    # Loss functions
    criterion_reg = torch.nn.MSELoss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for X, y in progress_bar:
        X = X.to(device)
        y_return = y['return'].to(device)
        y_direction = y['direction'].to(device)
        y_volatility = y['volatility'].to(device)
        
        optimizer.zero_grad()
        
        preds = model(X)
        
        # Multi-task Loss Calculation
        loss_return = criterion_reg(preds['return'].squeeze(), y_return)
        loss_direction = criterion_cls(preds['direction'], y_direction)
        loss_volatility = criterion_reg(preds['volatility'].squeeze(), y_volatility)
        
        # Weighted sum of losses (can be tuned)
        loss = loss_return + loss_direction + loss_volatility
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion_reg = torch.nn.MSELoss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    
    correct_direction = 0
    total_samples = 0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validation"):
            X = X.to(device)
            y_return = y['return'].to(device)
            y_direction = y['direction'].to(device)
            y_volatility = y['volatility'].to(device)
            
            preds = model(X)
            
            loss_return = criterion_reg(preds['return'].squeeze(), y_return)
            loss_direction = criterion_cls(preds['direction'], y_direction)
            loss_volatility = criterion_reg(preds['volatility'].squeeze(), y_volatility)
            
            loss = loss_return + loss_direction + loss_volatility
            total_loss += loss.item()
            
            # Accuracy metric
            pred_classes = torch.argmax(preds['direction'], dim=1)
            correct_direction += (pred_classes == y_direction).sum().item()
            total_samples += y_direction.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_direction / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, path, optimizer=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model weights loaded from {path}")
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint.get('epoch', 0)
