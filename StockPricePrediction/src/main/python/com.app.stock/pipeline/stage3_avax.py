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

def run_stage3():
    print("--- Stage 3: AVAX Specialization ---")
    
    # 1. Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    SEQ_LENGTH = 128
    
    # 2. Prepare Data (AVAX Specific)
    processor = DataProcessor(seq_length=SEQ_LENGTH)
    
    # Fetch AVAX Data
    # Ideally, we would also fetch funding rates/Open Interest here if available via yfinance or another API
    # For now, we stick to the yfinance data available
    df = processor.fetch_data(["AVAX-USD"], start_date="2020-09-01") # AVAX launched later
    
    df = processor.add_technical_indicators(df)
    X, y = processor.prepare_training_data(df, is_training=True)
    
    # Split
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train = {k: v[:train_size] for k, v in y.items()}
    y_val = {k: v[train_size:] for k, v in y.items()}
    
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X.shape[2]
    
    # 3. Initialize Model
    model = AVAXPredictor(input_dim=input_dim).to(DEVICE)
    
    # 4. Load Crypto Checkpoint
    checkpoint_path = "MachineLearningCode/StockPricePrediction/src/main/python/com.app.stock/checkpoints/Encoder_CRYPTO.pth"
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
    else:
        print(f"CRITICAL WARNING: Stage 2 checkpoint not found at {checkpoint_path}.")
        print("Ensure you have run stage2_crypto.py first.")
    
    # --- Phase A: Train Heads Only ---
    print("
[Phase A] Training Prediction Heads Only (High LR)")
    
    # Freeze Entire Encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    optimizer_heads = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(3): # Short phase to align heads
        print(f"Phase A Epoch {epoch+1}")
        train_loss = train_one_epoch(model, train_loader, optimizer_heads, DEVICE)
        val_loss, _ = validate(model, val_loader, DEVICE)
        print(f"Loss: {train_loss:.4f}")

    # --- Phase B: Fine-Tuning ---
    print("
[Phase B] Fine-Tuning Last Blocks (Low LR)")
    
    # Unfreeze LSTM and Last Transformer Layer
    for param in model.encoder.lstm.parameters():
        param.requires_grad = True
    for param in model.encoder.transformer.layers[-1].parameters():
        param.requires_grad = True
        
    # Extremely Low LR
    optimizer_finetune = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
    
    EPOCHS_B = 10
    best_val_loss = float('inf')
    save_path = "MachineLearningCode/StockPricePrediction/src/main/python/com.app.stock/checkpoints/AVAX_Predictor_Model.pth"
    
    for epoch in range(EPOCHS_B):
        print(f"Phase B Epoch {epoch+1}/{EPOCHS_B}")
        train_loss = train_one_epoch(model, train_loader, optimizer_finetune, DEVICE)
        val_loss, val_acc = validate(model, val_loader, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Direction Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer_finetune, epoch, save_path)
            
    print("Stage 3 Complete. Final Model Saved.")

if __name__ == "__main__":
    run_stage3()
