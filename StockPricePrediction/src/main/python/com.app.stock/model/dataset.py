import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional

class DataProcessor:
    def __init__(self, seq_length: int = 128, prediction_window: int = 1):
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.scaler = StandardScaler()
        self.fitted = False

    def fetch_data(self, tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetches historical data for multiple tickers."""
        print(f"Fetching data for: {tickers}...")
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        
        # Handle single ticker case where multi-index might not be created as expected
        if len(tickers) == 1:
            # Reformat to match multi-ticker structure or just process directly
            # For consistency, we'll assume we process one ticker at a time in the pipeline usually
            pass
            
        return data

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators matching the project standards.
        Expected input: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
        """
        df = df.copy()
        
        # Ensure we have the right columns (yfinance auto_adjust=True gives Open, High, Low, Close, Volume)
        # If columns are MultiIndex (Ticker, Field), this needs to be handled before calling this function
        
        # 1. Returns and Log Returns
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Trend Indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['EMA_20'] = ta.ema(df['Close'], length=20)
        
        # 3. Momentum Indicators
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_Signal'] = macd['MACDs_12_26_9']
        
        df['ROC_10'] = ta.roc(df['Close'], length=10)
        df['MOM_5'] = ta.mom(df['Close'], length=5)
        
        # 4. Volatility Indicators
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            df['BB_Upper'] = bb['BBU_20_2.0']
            df['BB_Lower'] = bb['BBL_20_2.0']
            
        df['Rolling_Std_20'] = df['Log_Return'].rolling(window=20).std()
        
        # 5. Temporal Features (Sine/Cosine encoding for cyclicity)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # 6. Lag Features (Short term history context at specific points)
        for lag in [1, 2, 3, 5]:
            df[f'Log_Return_Lag_{lag}'] = df['Log_Return'].shift(lag)

        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)
        return df

    def prepare_training_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepares X (features) and y (targets) for training.
        Targets:
        - Return: Next period log return
        - Direction: 0 (Down), 1 (Neutral), 2 (Up)
        - Volatility: Next period squared return (proxy for volatility)
        """
        
        # Targets
        # Next period return
        target_return = df['Log_Return'].shift(-self.prediction_window)
        
        # Direction (Threshold can be tuned, e.g., 0.001 for neutral)
        threshold = 0.001
        target_direction = target_return.apply(
            lambda x: 2 if x > threshold else (0 if x < -threshold else 1)
        )
        
        # Volatility (proxy)
        target_volatility = target_return ** 2

        # Drop last rows where targets are NaN
        valid_indices = ~target_return.isna()
        df_features = df[valid_indices].copy()
        
        y = {
            'return': target_return[valid_indices].values.astype(np.float32),
            'direction': target_direction[valid_indices].values.astype(np.int64),
            'volatility': target_volatility[valid_indices].values.astype(np.float32)
        }

        # Feature Selection (Exclude target-like columns if any, keep numerical)
        feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']] 
        # Actually we usually keep OHLCV but maybe normalized. Let's keep all numeric columns.
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X_raw = df_features[feature_cols].values

        if is_training:
            self.scaler.fit(X_raw)
            self.fitted = True
        elif not self.fitted:
            raise ValueError("Scaler must be fitted on training data first.")

        X_scaled = self.scaler.transform(X_raw)
        
        # Create Sequences
        X_seq, y_seq_return, y_seq_dir, y_seq_vol = [], [], [], []
        
        for i in range(len(X_scaled) - self.seq_length):
            X_seq.append(X_scaled[i : i + self.seq_length])
            y_seq_return.append(y['return'][i + self.seq_length])
            y_seq_dir.append(y['direction'][i + self.seq_length])
            y_seq_vol.append(y['volatility'][i + self.seq_length])
            
        return np.array(X_seq), {
            'return': np.array(y_seq_return),
            'direction': np.array(y_seq_dir),
            'volatility': np.array(y_seq_vol)
        }

class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        self.X = torch.from_numpy(X).float()
        self.y_return = torch.from_numpy(y['return']).float()
        self.y_direction = torch.from_numpy(y['direction']).long()
        self.y_volatility = torch.from_numpy(y['volatility']).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {
            'return': self.y_return[idx],
            'direction': self.y_direction[idx],
            'volatility': self.y_volatility[idx]
        }
