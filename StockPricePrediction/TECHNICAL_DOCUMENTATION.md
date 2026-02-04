# Stock Price Prediction System - Technical Documentation

## 1. Project Overview

The **Stock Price Prediction System** is a comprehensive machine learning and data analysis framework designed to forecast stock market trends and prices. The project utilizes a variety of methodologies ranging from statistical time-series analysis (ARIMA/SARIMA) to machine learning regressors (XGBoost, Random Forest, Linear Regression) and deep learning models (LSTM with Attention mechanisms).

The system aims to address four key objectives:
1.  **Next-Day Forecasting:** Predicting the closing price for the next trading day.
2.  **Long-Term Forecasting:** projecting prices for the next week or month (30-day horizon).
3.  **Trend Classification:** Determining market movement direction (Up/Down).
4.  **Data Analysis:** In-depth exploratory data analysis (EDA) and visualization of stock behaviors.

---

## 2. Repository Structure

The codebase is organized into three primary logical units within `src/main/python/com.app.stock/`:

### **Core (`/core`)**
Contains the foundational analysis and visualization logic.
*   `Stock_Price_Prediction_Exploratoy_Data_Analysis.ipynb`: The main data pipeline. Handles data loading, cleaning (Wide-to-Long reshaping), missing value imputation, feature engineering (RSI, MACD, Bollinger Bands), and initial model testing.
*   `Stock_Price_Prediction_Data_Visualization.ipynb`: dedicated plotting suite using Matplotlib, Seaborn, and Plotly for visual insights (Candlestick charts, Correlation heatmaps, etc.).

### **Process (`/process`)**
Contains specialized modeling workflows for specific forecasting horizons.
*   `Stock_Price_Prediction_Forecasting(Next_Day's_Price).ipynb`: Implementation of regression models (Linear, RF, XGBoost) and Univariate/Multivariate LSTMs for short-term prediction.
*   `Stock_Price_Prediction_Forecasting(Next_Week_Month_Price).ipynb`: Long-term forecasting workflows using SARIMA, Prophet, and recursive XGBoost strategies. Includes Hybrid models (SARIMA + XGBoost residuals).
*   `Stock_Price_Prediction_Forecasting(Next_Week_Month_Price)_WITH_TIMING.ipynb`: Performance-benchmarked version of the long-term forecasting module.

### **Util (`/util`)**
*   `EPBA-13 (Stock_Price_Prediction-Initial).ipynb`: Utility script for fetching raw historical data using the `yfinance` API.

---

## 3. System Architecture & Data Flow

The system follows a standard data science pipeline tailored for financial time-series data.

### **3.1 Data Flow**
1.  **Ingestion:** Raw data is acquired via `yfinance` or loaded from static Excel/CSV datasets (e.g., S&P daily updates).
2.  **Preprocessing:**
    *   **Reshaping:** Converts data from "Wide" format (tickers as columns) to "Long" format (Ticker column + Date index) for processing.
    *   **Cleaning:** Handles missing values using Forward-Fill (`ffill`) and Backward-Fill (`bfill`).
    *   **Outlier Removal:** Filters anomalies using Z-score (primarily on Volume data).
3.  **Feature Engineering:** Generates technical indicators:
    *   **Trend:** Rolling Mean (SMA), Exponential Moving Average (EMA).
    *   **Momentum:** RSI, MACD, Rate of Change (ROC).
    *   **Volatility:** Bollinger Bands, Standard Deviation.
    *   **Lag Features:** `Close_lag1` to `Close_lag10` to capture temporal dependencies.
4.  **Modeling:**
    *   **Regression:** Linear Regression, Random Forest, XGBoost.
    *   **Time Series:** ARIMA/SARIMA (via `pmdarima` for auto-tuning), Facebook Prophet.
    *   **Deep Learning:** LSTM (Long Short-Term Memory) networks with Dropout and Custom Attention layers.
    *   **Hybrid:** Combining linear statistical models (SARIMA) with non-linear ML models (XGBoost) on residuals.
5.  **Evaluation:** Metrics include RMSE (Root Mean Squared Error) and R² Score.

---

## 4. Key Modules & Technical Details

### **A. Exploratory Data Analysis (EDA)**
*   **Input:** Raw Excel/CSV data with MultiIndex headers.
*   **Logic:**
    *   Flattens MultiIndex columns (e.g., `('Close', 'AAPL')` -> `AAPL_Close`).
    *   Melts dataframe to long format for granular analysis.
    *   Performs extensive sanity checks on data counts, ticker coverage, and date ranges.
    *   Generates target variables: `Target_Close` (Regression) and `Target_Trend` (Classification).

### **B. Forecasting Models**

#### **1. Traditional Machine Learning**
*   **Linear Regression:** Serves as a strong baseline, often outperforming complex tree models for trend following due to the linear nature of price drifts.
*   **XGBoost & Random Forest:** Implemented with `GridSearchCV` for hyperparameter tuning. Note: These require careful handling (e.g., `TimeSeriesSplit`) to avoid look-ahead bias and often struggle with non-stationary trends without extensive differencing.

#### **2. Statistical Time Series**
*   **Auto ARIMA/SARIMA:** Automatically selects optimal `(p,d,q)` and seasonal parameters based on AIC scores. Used for capturing seasonality and linear trends.
*   **Prophet:** Utilized for its robustness to missing data and ability to model multiple seasonality components (weekly, yearly).

#### **3. Deep Learning**
*   **LSTM:** Sequential models built with Keras.
    *   **Architecture:** Stacked LSTM layers with Dropout (0.2) to prevent overfitting.
    *   **Multivariate Support:** Uses multiple features (Price + Technical Indicators) for prediction.
    *   **Attention Mechanism:** Custom `AttentionLayer` implemented to weight specific time steps in the input sequence, allowing the model to focus on critical historical events.

### **C. Visualization**
*   **Matplotlib/Seaborn:** Used for static distribution plots, boxplots, and heatmaps.
*   **Plotly:** Used for interactive Candlestick charts and scatter plots to analyze Volume vs. Price relationships.

---

## 5. Developer Guide

### **Requirements**
The project relies on a diverse stack of Python libraries. Key dependencies include:
*   `pandas`, `numpy`, `scipy` (Version specific: `numpy==1.26.4`, `scipy==1.10.1` recommended to avoid compatibility issues).
*   `scikit-learn`, `xgboost`, `statsmodels`.
*   `tensorflow` (Keras).
*   `pmdarima` (for Auto ARIMA).
*   `prophet`.
*   `yfinance`.
*   `plotly`, `matplotlib`, `seaborn`.

### **Setup & Execution**
1.  **Environment:** The code is optimized for Google Colab but can run locally.
    *   *Note:* If running locally, remove `google.colab` imports and adjust file paths (currently pointing to `/content/drive/`).
2.  **Data:** Ensure the dataset (e.g., `stock_data_long_filled.csv`) is present in the specified directory.
3.  **Execution Order:**
    1.  Run `Stock_Price_Prediction_Exploratoy_Data_Analysis.ipynb` to clean data and understand feature distributions.
    2.  Run `Stock_Price_Prediction_Forecasting(Next_Day's_Price).ipynb` for short-term model training.
    3.  Run the `Next_Week_Month` notebooks for long-horizon planning.

---

## 6. Code Quality & Risks

### **Potential Risks & Anti-Patterns**
1.  **Hardcoded Paths:** File paths are hardcoded to Google Drive locations (`/content/drive/My Drive/...`). This breaks portability.
    *   *Recommendation:* Use relative paths or environment variables for configuration.
2.  **Dependency Hell:** The notebooks contain explicit `!pip install/uninstall` commands to fix version conflicts (specifically `numpy`, `scipy`, `pmdarima`).
    *   *Recommendation:* Use a `requirements.txt` or `environment.yml` lock file to manage this cleanly outside the code.
3.  **Data Leakage:** While `TimeSeriesSplit` is used in some sections, care must be taken with scaling. `MinMaxScaler` should ideally be fitted *only* on the training split, not the entire dataset before splitting, to simulate real-world conditions perfectly.
4.  **Error Handling:** Some data loading blocks assume specific Excel formats (header rows). If the input format changes, the pipeline will break.

### **Performance Considerations**
*   **Grid Search:** The `GridSearchCV` implementation for XGBoost and Random Forest can be computationally expensive on large datasets.
*   **LSTM Training:** Deep learning models are trained for up to 100 epochs. Ensure GPU acceleration is enabled when retraining.

### **Missing Elements**
*   **Unit Tests:** There are no explicit unit tests for the data transformation logic or custom Attention layer.
*   **Model Persistence:** There is no code to save (`pickle` or `.h5`) the trained models for deployment/inference. Models are trained and evaluated in-memory.
