import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb


def train_and_validate_models(test_size=0.2):
    """
    使用特征工程后的数据训练随机森林和 XGBoost 模型，并进行全面模型验证：
    1) 时间序列交叉验证
    2) 留出测试集评估
    3) 特征重要性可视化
    4) 预测 vs 实际图
    5) 残差分析
    """
    # 1. 读取特征数据
    BASE_DIR = Path(__file__).resolve().parent.parent
    path = BASE_DIR / "data" / "cleaned" / "air_quality_features.csv"

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"成功读取文件: {path}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {path}")
        print("请先运行 src/feature_engineering.py 来生成该文件。")
        return

    # 2. 选择特征列
    feature_cols = [
        'TEMP', 'PRES', 'Iws', 'day_of_week',  # 原始气象 + weekday
        'pm25_lag1', 'pm25_lag24', 'pm25_rolling_24',  # PM2.5特征
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',  # 周期性时间特征
        'Iws_lag6', 'TEMP_lag3', 'PRES_lag6'  # 气象滞后特征
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df['pm2.5']

    # ----------------------------
    # 3. 时间序列交叉验证
    # ----------------------------
    print("\n=== 时间序列交叉验证 ===")
    tscv = TimeSeriesSplit(n_splits=5)

    # RandomForestRegressor 模型
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf_cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='r2', n_jobs=-1)
    print(f"Random Forest 交叉验证 R² 均值: {rf_cv_scores.mean():.4f}")

    # XGBoost 模型（移除早期停止）
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring='r2', n_jobs=-1)
    print(f"XGBoost 交叉验证 R² 均值: {xgb_cv_scores.mean():.4f}")

    # ----------------------------
    # 4. 留出测试集评估
    # ----------------------------
    print("\n=== 留出测试集评估 ===")
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 训练随机森林
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)

    # 训练 XGBoost
    xgb_model.fit(X_train, y_train)  # 这里移除 early_stopping_rounds 参数
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, y_pred_xgb)
    xgb_mse = mean_squared_error(y_test, y_pred_xgb)
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)

    print(f"Random Forest 测试集 R²: {rf_r2:.4f}")
    print(f"Random Forest 测试集 MSE: {rf_mse:.2f}")
    print(f"Random Forest 测试集 MAE: {rf_mae:.2f}")

    print(f"XGBoost 测试集 R²: {xgb_r2:.4f}")
    print(f"XGBoost 测试集 MSE: {xgb_mse:.2f}")
    print(f"XGBoost 测试集 MAE: {xgb_mae:.2f}")

    # ----------------------------
    # 5. 特征重要性
    # ----------------------------
    print("\n=== 特征重要性 ===")
    rf_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values()

    plt.figure(figsize=(10, 6))
    rf_importances.plot(kind='barh', color='teal', label="Random Forest")
    xgb_importances.plot(kind='barh', color='orange', label="XGBoost")
    plt.title("Feature Importance (Random Forest vs XGBoost)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 6. 预测 vs 实际图
    # ----------------------------
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Actual PM2.5', color='black', alpha=0.7)
    plt.plot(y_test.index, y_pred_rf, label='Random Forest Predicted PM2.5', color='red', linestyle='--')
    plt.plot(y_test.index, y_pred_xgb, label='XGBoost Predicted PM2.5', color='green', linestyle='--')
    plt.title("PM2.5 Prediction: Actual vs Predicted (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 7. 残差分析
    # ----------------------------
    rf_residuals = y_test - y_pred_rf
    xgb_residuals = y_test - y_pred_xgb

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, rf_residuals, label='RF Residuals', color='red')
    plt.plot(y_test.index, xgb_residuals, label='XGB Residuals', color='green')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Residuals (Actual - Predicted) - RF vs XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_validate_models()
