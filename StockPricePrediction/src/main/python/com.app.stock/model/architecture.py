import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, n_head=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 1. Temporal Convolutional Layers (TCN)
        # Captures short-term patterns and reduces noise
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Transformer Encoder Blocks
        # Captures long-range dependencies and market regimes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Optional LSTM Smoothing Layer
        # Sequential memory continuity
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (Batch, TimeSteps, Features)
        
        # Project to d_model
        x = self.input_proj(x)
        
        # TCN Forward
        # Conv1d expects (Batch, Channels, Length)
        x_tcn = x.transpose(1, 2)
        x_tcn = self.tcn(x_tcn)
        x_tcn = x_tcn.transpose(1, 2)
        
        # Residual connection from input projection to TCN output
        x = self.layer_norm(x + x_tcn)
        
        # Transformer Forward
        x_trans = self.transformer(x)
        
        # LSTM Forward
        # LSTM returns (output, (h_n, c_n))
        x_lstm, _ = self.lstm(x_trans)
        
        # We take the latent representation of the LAST timestep
        # representing the state of the market at time T
        latent_vector = x_lstm[:, -1, :]
        
        return latent_vector

class AVAXPredictor(nn.Module):
    def __init__(self, input_dim, d_model=128, n_head=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = HybridEncoder(input_dim, d_model, n_head, n_layers, dropout)
        
        # Multi-Task Prediction Heads
        
        # 1. Return Regression Head
        self.head_return = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # 2. Direction Classification Head (Up, Down, Neutral)
        self.head_direction = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3) 
        )
        
        # 3. Volatility/Risk Head (Must be positive)
        self.head_volatility = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        
        return {
            'return': self.head_return(latent),
            'direction': self.head_direction(latent),
            'volatility': self.head_volatility(latent)
        }
