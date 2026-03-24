"""
Train ConvLSTM on Preprocessed SEIR Data
=========================================

Loads preprocessed SEIR epidemic data and trains a ConvLSTM model
to predict future infection maps from past observations.

Input:  (batch, seq_len=8, channels=1, 16, 16)  — 8 days of I data
Output: (batch, pred_len=4, channels=1, 16, 16)  — next 4 days prediction

Usage:
    python train_convlstm.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory so we can import the preprocessing utilities
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from seir_preprocessing import load_preprocessed_data, DataNormalizer
from ConvLSTM import ConvLSTMLayers


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class TrainConfig:
    """Training hyperparameters."""
    # Data
    PREPROCESSED_DIR = os.path.join(parent_dir, 'preprocessed_output')
    
    # Model
    INPUT_CHANNELS = 1          # Single channel (I compartment)
    HIDDEN_CHANNELS = [256, 256]  # Two ConvLSTM layers with 64 hidden channels
    KERNEL_SIZES = [3, 3]       # 3x3 kernels
    NUM_LAYERS = 2
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    WEIGHT_DECAY = 1e-5
    
    # Scheduler
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Output
    SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class ConvLSTMForecaster(nn.Module):
    """
    ConvLSTM-based forecasting model.
    
    Takes seq_len frames as input and predicts pred_len frames.
    Architecture:
        ConvLSTMLayers (2 layers) → 1x1 Conv projection → output
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_sizes,
                 num_layers, pred_len):
        super().__init__()
        
        self.pred_len = pred_len
        
        # ConvLSTM encoder
        self.encoder = ConvLSTMLayers(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_sizes,
            num_layers=num_layers,
            bias=True
        )
        
        # Projection: map last hidden state to pred_len output frames
        # Last layer hidden_channels → pred_len * input_channels
        last_hidden = hidden_channels[-1]
        self.projection = nn.Conv2d(
            last_hidden, pred_len * input_channels,
            kernel_size=1, bias=True
        )
        
        self.input_channels = input_channels
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels, height, width)
        Returns:
            pred: (batch, pred_len, channels, height, width)
        """
        batch_size = x.size(0)
        height, width = x.size(3), x.size(4)
        
        # Run through ConvLSTM layers
        layer_outputs, last_states = self.encoder(x)
        
        # Use the last hidden state from the last layer
        h_last = last_states[-1][0]  # (batch, hidden_channels, H, W)
        
        # Project to pred_len output frames
        out = self.projection(h_last)  # (batch, pred_len * channels, H, W)
        
        # Reshape to (batch, pred_len, channels, H, W)
        out = out.view(batch_size, self.pred_len, self.input_channels,
                       height, width)
        
        return out


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        
        total_loss += loss.item()
        num_batches += 1
        
        all_preds.append(pred.cpu())
        all_targets.append(batch_y.cpu())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    mse = nn.functional.mse_loss(all_preds, all_targets).item()
    mae = nn.functional.l1_loss(all_preds, all_targets).item()
    rmse = mse ** 0.5
    
    return avg_loss, {'mse': mse, 'mae': mae, 'rmse': rmse}


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg = TrainConfig()
    
    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Load preprocessed data ---
    print(f"\nLoading preprocessed data from: {cfg.PREPROCESSED_DIR}")
    data = load_preprocessed_data(cfg.PREPROCESSED_DIR)
    
    arrays = data['arrays']
    normalizer = data['normalizer']
    config = data['config']
    
    print(f"  Config: seq_len={config['seq_len']}, pred_len={config['pred_len']}, "
          f"norm={config['norm_method']}")
    
    X_train = torch.from_numpy(arrays['X_train']).float()
    Y_train = torch.from_numpy(arrays['Y_train']).float()
    X_test  = torch.from_numpy(arrays['X_test']).float()
    Y_test  = torch.from_numpy(arrays['Y_test']).float()
    
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Y_test:  {Y_test.shape}")
    
    # Handle optional validation set
    has_val = 'X_val' in arrays and arrays['X_val'] is not None
    if has_val:
        X_val = torch.from_numpy(arrays['X_val']).float()
        Y_val = torch.from_numpy(arrays['Y_val']).float()
        print(f"  X_val:   {X_val.shape}")
    
    # --- Create DataLoaders ---
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=cfg.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=cfg.BATCH_SIZE, shuffle=False
    )
    if has_val:
        val_loader = DataLoader(
            TensorDataset(X_val, Y_val),
            batch_size=cfg.BATCH_SIZE, shuffle=False
        )
    
    # --- Build model ---
    pred_len = config['pred_len']
    model = ConvLSTMForecaster(
        input_channels=cfg.INPUT_CHANNELS,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        kernel_sizes=cfg.KERNEL_SIZES,
        num_layers=cfg.NUM_LAYERS,
        pred_len=pred_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ConvLSTMForecaster")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Hidden channels:  {cfg.HIDDEN_CHANNELS}")
    print(f"  Prediction steps: {pred_len}")
    
    # --- Optimizer, criterion, scheduler ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                           weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.SCHEDULER_PATIENCE,
        factor=cfg.SCHEDULER_FACTOR
    )
    
    # --- Training loop ---
    print(f"\n{'='*60}")
    print(f"TRAINING ({cfg.EPOCHS} epochs, batch_size={cfg.BATCH_SIZE})")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        
        # Validate
        eval_loader = val_loader if has_val else test_loader
        eval_name = "Val" if has_val else "Test"
        val_loss, val_metrics = evaluate(model, eval_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{cfg.EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | "
              f"{eval_name} Loss: {val_loss:.6f} | "
              f"RMSE: {val_metrics['rmse']:.6f} | "
              f"MAE: {val_metrics['mae']:.6f} | "
              f"LR: {lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(cfg.SAVE_DIR, 'best_convlstm.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': {
                    'input_channels': cfg.INPUT_CHANNELS,
                    'hidden_channels': cfg.HIDDEN_CHANNELS,
                    'kernel_sizes': cfg.KERNEL_SIZES,
                    'num_layers': cfg.NUM_LAYERS,
                    'pred_len': pred_len,
                }
            }, save_path)
            print(f"  → Saved best model (loss={val_loss:.6f})")
    
    # --- Final evaluation on test set ---
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(os.path.join(cfg.SAVE_DIR, 'best_convlstm.pth'),
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"  Test MSE:  {test_metrics['mse']:.6f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"  Test MAE:  {test_metrics['mae']:.6f}")
    print(f"  Best epoch: {checkpoint['epoch']}")
    
    # Save final metrics
    metrics_path = os.path.join(cfg.SAVE_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_epoch': checkpoint['epoch'],
            'test_mse': test_metrics['mse'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'best_val_loss': best_val_loss,
            'config': checkpoint['config'],
        }, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")
    print(f"  Model saved to:   {os.path.join(cfg.SAVE_DIR, 'best_convlstm.pth')}")


if __name__ == '__main__':
    main()
