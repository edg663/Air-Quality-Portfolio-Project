
# Beijing Air Quality Prediction (PM2.5)

**Time Series Forecasting & Machine Learning Portfolio**  
*(北京空气质量预测项目 — 完整机器学习时间序列分析流程)*

---

## 📘 Overview

This project presents an end‑to‑end machine learning workflow for predicting **PM2.5 concentration levels** in Beijing using the **UCI PRSA dataset (2010–2014)**.  
It highlights advanced **time‑series feature engineering**, tree‑based ensemble modeling, rigorous **TimeSeries Cross‑Validation**, and visualization-driven interpretation.

This project was developed as part of an academic portfolio for research and upper‑year project applications.

---

## 🚀 Key Features (What Makes This Project Strong)

### 🔹 1. Advanced Time‑Series Feature Engineering  
Implemented in `src/feature_engineering.py`:

- **Lag Features** (`pm25_lag1`, `pm25_lag24`) → capture short-term & daily temporal dependence  
- **Rolling Window Features** → 24‑hour smoothed trends  
- **Cyclical Encoding**  
  Converts hour / month / weekday into sine‑cosine components to keep continuity  
  (`23 → 0` becomes close on a circle)  
- **Meteorological Historical Lags** for temperature, humidity, wind speed…

### 🔹 2. Robust Ensemble Modeling  
Implemented in `modeling_rf.py` & `modeling_xgb.py`:

- **Random Forest Regressor**
- **XGBoost Regressor** (best model)
- Trains with **TimeSeriesSplit** (prevents data leakage)

### 🔹 3. Clear Interpretation & Analysis

- Feature Importance (built‑in RF & XGB)
- Actual vs Predicted line charts
- Residual distribution analysis

---

## 📂 Project Structure

```
Air-Quality-Portfolio-Project/
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── images/                 # Plots (feature importance, predictions)
├── models/                 # Saved trained models
│
├── notebooks/
│   └── AirQuality_Report_CNfont_Improved.ipynb
│
├── src/
│   ├── test_data_cleaning.py
│   ├── feature_engineering.py
│   ├── modeling_rf.py
│   ├── modeling_xgb.py
│   └── visualization_pm25.py
│
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

- **Python 3.9+**
- Pandas / NumPy
- scikit‑learn
- XGBoost
- Matplotlib / Seaborn
- Jupyter Notebook

---

## ⚙️ Methodology

### 1. Data Cleaning  
File: `test_data_cleaning.py`

- Linear interpolation for missing timestamps  
- Drop incomplete rows  
- Basic quality control checks

---

### 2. Feature Engineering  
File: `feature_engineering.py`

#### ✔ Lag Features  
Captures autocorrelation patterns:

```
pm25_lag1   # previous hour
pm25_lag24  # previous day same hour
```

#### ✔ Rolling Window Features  
Smooths noisy short-term variance.

#### ✔ Cyclical Encoding  
Example (Hour → sin/cos):

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

Ensures 23:00 and 00:00 remain close in encoded feature space.

---

### 3. Model Training & Evaluation  
Files:  
- `modeling_rf.py`  
- `modeling_xgb.py`

#### Validation:
Uses **5‑fold TimeSeriesSplit** to ensure no future leakage.

#### Metrics:
- RMSE
- MAE
- R² score

---

## 📊 Performance Summary

| Model | Feature Set | Test R² | Notes |
|------|-------------|---------|-------|
| **Linear Regression** | Weather only | ~0.11 | Strong underfitting |
| **Quadratic Temp Model** | Weather + Temp² | ~0.12 | Very small gain |
| **Random Forest** | Lags + Cyclical | **~0.94** | Strong performance |
| **XGBoost** | Lags + Cyclical | **~0.945** | Best model |

---

## 📈 Key Plots (in `images/`)

- Time‑series prediction vs actual PM2.5  
- Residual distribution  
- Model comparison  
- Feature importance (RF & XGB)

---

## 🧪 Quick Start

### Step 1 — Install dependencies  
```
pip install -r requirements.txt
```

### Step 2 — Clean data  
```
python src/test_data_cleaning.py
```

### Step 3 — Generate engineered features  
```
python src/feature_engineering.py
```

### Step 4 — Train baseline + ensemble models  
```
python src/modeling_rf.py
```

### Step 5 — Train production XGBoost model  
```
python src/modeling_xgb.py
```

---

## 🔮 Future Improvements

- Hyperparameter tuning (Optuna / GridSearchCV)
- Add LightGBM & CatBoost
- Train deep learning models (LSTM / GRU)
- Deploy a Streamlit dashboard
- Integrate SHAP for advanced model explainability

---

## 👤 Author

**edg663**  
Educational Portfolio Project  
GitHub: https://github.com/edg663/Air-Quality-Portfolio-Project
