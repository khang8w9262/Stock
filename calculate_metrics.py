import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_percentage_error,
                           mean_squared_error,
                           mean_absolute_error,
                           r2_score)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import các hàm cần thiết từ back.py
from back import (load_data, create_advanced_features, simulate_future_price,
                 build_lightgbm_model, load_model, save_model, load_scaler, save_scaler,
                 find_baodautu_prediction)

def calculate_and_save_metrics():
    """
    Tính toán và lưu trữ metrics cho HNX-Index với các khoảng thời gian cụ thể
    """
    # Thiết lập đường dẫn
    train_dir = 'Train'
    ve_dir = 'Ve'
    metrics_dir = 'metrics'
    
    # Tạo thư mục metrics nếu chưa tồn tại
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    print("\nBắt đầu tính toán metrics cho HNX-Index...")
    
    try:
        # Chỉ xử lý HNX-Index
        stock_name = "HNX-Index"
        file_path_train = os.path.join(train_dir, f"{stock_name}.csv")
        file_path_display = os.path.join(ve_dir, f"{stock_name}_TT.csv")
        
        if not os.path.exists(file_path_train) or not os.path.exists(file_path_display):
            print(f"Không tìm thấy file dữ liệu cho {stock_name}")
            return
        
        # Khởi tạo dict để lưu metrics
        model_metrics = {
            'normal': {},
            'future': {},
            'supplement': {}
        }
        
        # Đọc và xử lý dữ liệu
        data_train = load_data(file_path_train)
        data_display = load_data(file_path_display)
        
        if data_train.empty or data_display.empty:
            print(f"Không thể đọc dữ liệu cho {stock_name}")
            return
        
        # Định nghĩa các khoảng thời gian
        normal_start = pd.to_datetime('13/03/2024', format='%d/%m/%Y')
        normal_end = pd.to_datetime('26/03/2024', format='%d/%m/%Y')
        future_start = pd.to_datetime('13/03/2025', format='%d/%m/%Y')
        future_end = pd.to_datetime('26/03/2025', format='%d/%m/%Y')
        
        # Lọc bỏ ngày cuối tuần
        data_train = data_train[~data_train['Date'].dt.dayofweek.isin([5, 6])]
        data_display = data_display[~data_display['Date'].dt.dayofweek.isin([5, 6])]
        
        # Tạo features
        data_lagged = create_advanced_features(data_train, n_lags=10)
        
        if data_lagged.empty:
            print("Không thể tạo features")
            return
        
        # Chỉ sử dụng các lag features
        lag_features = [f'lag_{i}' for i in range(1, 11)]
        features = [col for col in lag_features if col in data_lagged.columns]
        
        if not all(feature in data_lagged.columns for feature in features):
            print("Không đủ lag features")
            return
        
        X = data_lagged[features]
        y = data_lagged['Price']
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Khởi tạo models
        models = {
            'rf': RandomForestRegressor(
                n_estimators=2000,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                objective='reg:squarederror',
                colsample_bytree=0.8,
                learning_rate=0.02,
                max_depth=8,
                alpha=2,
                n_estimators=2000,
                subsample=0.8,
                min_child_weight=3
            ),
            'lgb': build_lightgbm_model({
                'num_leaves': 127,
                'learning_rate': 0.02,
                'feature_fraction': 0.8,
                'n_estimators': 2000,
                'min_child_samples': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }),
            'dt': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
                random_state=42
            )
        }
        
        # Xử lý cho từng model
        for model_name, model in models.items():
            print(f"\nXử lý model {model_name}...")
            
            # 1. Tính metrics cho dự báo thông thường (13/03/2024 - 26/03/2024)
            normal_mask = (data_display['Date'] >= normal_start) & (data_display['Date'] <= normal_end)
            normal_data = data_display[normal_mask]
            
            if not normal_data.empty:
                model.fit(X_scaled, y)
                normal_pred = model.predict(X_scaled[normal_mask])
                
                model_metrics['normal'][model_name] = {
                    'MSE': mean_squared_error(normal_data['Price'], normal_pred),
                    'MAE': mean_absolute_error(normal_data['Price'], normal_pred),
                    'RMSE': np.sqrt(mean_squared_error(normal_data['Price'], normal_pred)),
                    'MAPE': mean_absolute_percentage_error(normal_data['Price'], normal_pred),
                    'R^2': r2_score(normal_data['Price'], normal_pred)
                }
            
            # 2. Tính metrics cho dự báo tương lai (13/03/2025 - 26/03/2025)
            last_features = X[-1:].values
            future_dates = pd.date_range(start=future_start, end=future_end, freq='B')
            future_predictions = simulate_future_price(model, last_features, scaler, future_dates, stock_name=stock_name)
            
            if future_predictions:
                predictions = [p[1] for p in future_predictions]
                if len(predictions) > 1:
                    returns = np.diff(predictions) / predictions[:-1]
                    model_metrics['future'][model_name] = {
                        'MSE': np.sum(returns > 0) / len(returns),
                        'MAE': np.std(returns) * np.sqrt(252),
                        'RMSE': abs((predictions[-1] - predictions[0]) / predictions[0]),
                        'MAPE': abs(np.mean(returns) * 100),
                        'R^2': np.mean(returns) / (np.std(returns) if np.std(returns) != 0 else 1)
                    }
            
            # 3. Tính metrics cho dự báo bổ sung (13/03/2025 - 26/03/2025)
            baodautu_dates = []
            baodautu_prices = []
            
            for date in future_dates:
                prediction, is_from_file = find_baodautu_prediction(stock_name, date)
                if prediction is not None:
                    baodautu_dates.append(date)
                    baodautu_prices.append(prediction)
            
            if baodautu_dates and future_predictions:
                combined_predictions = []
                for i, date in enumerate(future_dates):
                    if date in baodautu_dates:
                        idx = baodautu_dates.index(date)
                        baodautu_price = baodautu_prices[idx]
                        model_price = predictions[i]
                        combined_price = 0.7 * baodautu_price + 0.3 * model_price
                        combined_predictions.append(combined_price)
                
                if len(combined_predictions) > 1:
                    returns = np.diff(combined_predictions) / combined_predictions[:-1]
                    model_metrics['supplement'][model_name] = {
                        'MSE': np.sum(returns > 0) / len(returns),
                        'MAE': np.std(returns) * np.sqrt(252),
                        'RMSE': abs((combined_predictions[-1] - combined_predictions[0]) / combined_predictions[0]),
                        'MAPE': abs(np.mean(returns) * 100),
                        'R^2': np.mean(returns) / (np.std(returns) if np.std(returns) != 0 else 1)
                    }
        
        # Lưu kết quả vào file
        metrics_file = os.path.join(metrics_dir, 'system_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(model_metrics, f, indent=4)
        
        print(f"\nĐã lưu metrics vào file: {metrics_file}")
        print("\nKết quả metrics:")
        print(json.dumps(model_metrics, indent=4))
        
    except Exception as e:
        print(f"\nLỗi khi tính toán metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_and_save_metrics() 