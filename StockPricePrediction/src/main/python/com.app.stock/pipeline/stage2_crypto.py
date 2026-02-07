import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../..')))

from MachineLearningCode.StockPricePrediction.src.main.python.com.app.stock.model.dataset import DataProcessor, StockDataset
from MachineLearningCode.StockPricePrediction.src.main.python.com.app.stock.model.architecture import AVAXPredictor
from MachineLearningCode.StockPricePrediction.src.main.python.com.app.stock.model.train_utils import train_one_epoch, validate, save_checkpoint, load_checkpoint

def run_stage2():
    print("--- Stage 2: Multi-Crypto Domain Adaptation ---")
    
    # 1. Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 10 # Adjust as needed
    LR = 1e-5   # Low learning rate for fine-tuning
    SEQ_LENGTH = 128
    FEATURES = 40 # Approximation, depends on dataset.py output
    
    # 2. Prepare Data
    processor = DataProcessor(seq_length=SEQ_LENGTH)
    
    # Fetch Crypto Data
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']
    df = processor.fetch_data(tickers, start_date="2020-01-01")
    
    # Process each ticker and concatenate or train sequentially
    # For simplicity, we'll melt or just process one big list of sequences
    # A better approach for domain adaptation: Mix sequences from all assets
    
    all_X = []
    all_y = {'return': [], 'direction': [], 'volatility': []}
    
    # We need to process each ticker separately to avoid jumps in price between tickers
    # fetch_data with group_by='ticker' returns a MultiIndex dataframe if multiple tickers
    
    if len(tickers) > 1:
        # Flatten MultiIndex
        for ticker in tickers:
            df_ticker = df[ticker].dropna()
            df_ticker = processor.add_technical_indicators(df_ticker)
            X, y = processor.prepare_training_data(df_ticker, is_training=True)
            all_X.append(X)
            all_y['return'].append(y['return'])
            all_y['direction'].append(y['direction'])
            all_y['volatility'].append(y['volatility'])
            
        import numpy as np
        X_final = np.concatenate(all_X)
        y_final = {
            'return': np.concatenate(all_y['return']),
            'direction': np.concatenate(all_y['direction']),
            'volatility': np.concatenate(all_y['volatility'])
        }
    else:
        df = processor.add_technical_indicators(df)
        X_final, y_final = processor.prepare_training_data(df, is_training=True)

    # Split Train/Val (Chronological)
    train_size = int(len(X_final) * 0.8)
    X_train, X_val = X_final[:train_size], X_final[train_size:]
    y_train = {k: v[:train_size] for k, v in y_final.items()}
    y_val = {k: v[train_size:] for k, v in y_final.items()}
    
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Update feature dimension based on actual data
    input_dim = X_final.shape[2]
    print(f"Input Feature Dimension: {input_dim}")
    
    # 3. Initialize Model
    model = AVAXPredictor(input_dim=input_dim).to(DEVICE)
    
    # 4. Load S&P 500 Checkpoint
    checkpoint_path = "MachineLearningCode/StockPricePrediction/src/main/python/com.app.stock/checkpoints/Encoder_SP500.pth"
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
    else:
        print(f"WARNING: Stage 1 checkpoint not found at {checkpoint_path}.")
        print("Starting with random weights (Not recommended for proper Transfer Learning).")
    
    # 5. Freeze Layers (TCN + 1st Transformer Layer)
    print("Freezing TCN and first Transformer layer...")
    for param in model.encoder.tcn.parameters():
        param.requires_grad = False
        
    # Accessing the first layer of TransformerEncoder
    # model.encoder.transformer is a TransformerEncoder
    # .layers is the ModuleList
    for param in model.encoder.transformer.layers[0].parameters():
        param.requires_grad = False
        
    # 6. Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    # 7. Training Loop
    best_val_loss = float('inf')
    save_path = "MachineLearningCode/StockPricePrediction/src/main/python/com.app.stock/checkpoints/Encoder_CRYPTO.pth"
    
    for epoch in range(EPOCHS):
        print(f"
Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Direction Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, save_path)
            
    print("Stage 2 Complete. Checkpoint saved.")

if __name__ == "__main__":
    run_stage2()
