import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


def train_xgboost_model():
    """
    使用 XGBoost 训练回归模型，并评估其表现。
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_path = BASE_DIR / "data" / "cleaned" / "air_quality_features.csv"

    # 读取数据
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    # 转换 'cbwd' 列为独热编码 (如果 'cbwd' 是类别特征)
    if 'cbwd' in df.columns:
        df = pd.get_dummies(df, columns=['cbwd'], drop_first=True)

    # 特征和目标变量
    X = df.drop(columns=["pm2.5"])  # 假设目标变量是 'pm2.5'
    y = df["pm2.5"]

    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建 XGBoost 模型
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,  # 防止过拟合
        n_jobs=-1,
        random_state=42
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # 预测结果
    y_pred = model.predict(X_test)

    # 模型评估
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost R²: {r2:.4f}")

    # 保存模型 (确保目标文件夹存在)
    model_output_path = BASE_DIR / "models" / "xgboost_model.pkl"

    # 确保目标文件夹存在
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_output_path)
    print(f"模型已保存至: {model_output_path}")


if __name__ == "__main__":
    train_xgboost_model()
