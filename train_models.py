import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from back import load_data, create_advanced_features

def optimize_rf(X_train, y_train):
    """Optimize RandomForest hyperparameters"""
    param_grid = {
        'n_estimators': [1000, 1500, 2000],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True],
        'oob_score': [True],
        'warm_start': [True]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print("Best RandomForest parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def optimize_xgb(X_train, y_train):
    """Optimize XGBoost hyperparameters"""
    param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05],
        'n_estimators': [1000, 1500, 2000],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.1, 1.0, 5.0],
        'tree_method': ['hist'],
        'booster': ['gbtree']
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print("Best XGBoost parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def optimize_lgb(X_train, y_train):
    """Optimize LightGBM hyperparameters"""
    param_grid = {
        'num_leaves': [31, 63, 127],
        'max_depth': [10, 15, 20],
        'learning_rate': [0.01, 0.03, 0.05],
        'n_estimators': [1000, 1500, 2000],
        'min_child_samples': [5, 10, 20],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5],
        'boosting_type': ['gbdt'],
        'extra_trees': [True],
        'path_smooth': [0.1]
    }
    
    lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        estimator=lgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print("Best LightGBM parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def optimize_dt(X_train, y_train):
    """Optimize Decision Tree hyperparameters"""
    param_grid = {
        'max_depth': [8, 10, 12],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'splitter': ['best', 'random'],
        'criterion': ['squared_error', 'friedman_mse']
    }
    
    dt = DecisionTreeRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print("Best Decision Tree parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def train_and_save_models(stock_name, train_file):
    """Train and save optimized models for a stock"""
    print(f"\nTraining models for {stock_name}...")
    
    # Load and preprocess data
    data = load_data(train_file)
    if data is None or data.empty:
        print(f"Error: Could not load data for {stock_name}")
        return False
        
    # Create features
    features = create_advanced_features(data)
    if features is None or features.empty:
        print(f"Error creating features for {stock_name}")
        return False
        
    # Add more advanced features
    features = add_advanced_features(features)
    
    # Get feature columns for training
    feature_columns = [col for col in features.columns if col not in ['Date', 'target']]
    
    # Split features and target
    X = features[feature_columns]
    y = features['target']
    
    # Split into train and validation sets using time series split
    tscv = TimeSeriesSplit(n_splits=5)
    best_models = {}
    best_scores = {}
    
    for model_name in ['rf', 'xgb', 'lgb', 'dt']:
        print(f"\nTraining {model_name.upper()} models...")
        best_score = float('-inf')
        best_model = None
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model with optimization
            if model_name == 'rf':
                model = optimize_rf(X_train_scaled, y_train)
            elif model_name == 'xgb':
                model = optimize_xgb(X_train_scaled, y_train)
            elif model_name == 'lgb':
                model = optimize_lgb(X_train_scaled, y_train)
            else:
                model = optimize_dt(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val_scaled)
            score = -mean_squared_error(y_val, y_pred)  # Negative MSE for maximization
            
            if score > best_score:
                best_score = score
                best_model = model
        
        best_models[model_name] = best_model
        best_scores[model_name] = best_score
    
    # Save best models and scaler
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Final training on full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for model_name, model in best_models.items():
        # Retrain on full dataset
        model.fit(X_scaled, y)
        # Save model
        model_path = os.path.join(models_dir, f'{stock_name}_{model_name}_model.joblib')
        joblib.dump(model, model_path)
        print(f"Saved {model_name} model to {model_path}")
    
    # Save feature order and scaler
    joblib.dump(feature_columns, os.path.join(models_dir, f'{stock_name}_feature_order.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, f'{stock_name}_scaler.joblib'))
    
    print("\nBest validation scores:")
    for model_name, score in best_scores.items():
        print(f"{model_name.upper()}: {-score:.6f} MSE")
    
    return True

def add_advanced_features(df):
    """Add more advanced technical indicators and features"""
    # Copy dataframe to avoid modifying original
    df = df.copy()
    
    # Price momentum features
    df['Price_Mom_1'] = df['Price'].pct_change(1)
    df['Price_Mom_5'] = df['Price'].pct_change(5)
    df['Price_Mom_10'] = df['Price'].pct_change(10)
    
    # Exponential moving averages
    df['EMA_5_10_Cross'] = df['EMA_5'] - df['EMA_10']
    df['EMA_10_20_Cross'] = df['EMA_10'] - df['EMA_20']
    
    # Volatility features
    df['ATR'] = df['High'] - df['Low']
    df['ATR_MA_5'] = df['ATR'].rolling(window=5).mean()
    df['ATR_MA_10'] = df['ATR'].rolling(window=10).mean()
    
    # Volume features
    df['Volume_Mom_1'] = df['Volume'].pct_change(1)
    df['Volume_Mom_5'] = df['Volume'].pct_change(5)
    df['Volume_Price_Trend'] = df['Volume'] * df['Price_Pct_Change']
    
    # RSI variations
    df['RSI_MA_5'] = df['RSI'].rolling(window=5).mean()
    df['RSI_MA_10'] = df['RSI'].rolling(window=10).mean()
    
    # MACD variations
    df['MACD_Signal_Cross'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Zero_Cross'] = df['MACD']
    
    # Price pattern features
    df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    df['Higher_High_Count'] = df['Higher_High'].rolling(window=5).sum()
    df['Lower_Low_Count'] = df['Lower_Low'].rolling(window=5).sum()
    
    # Support and resistance features
    df['Support_Level'] = df['Low'].rolling(window=10).min()
    df['Resistance_Level'] = df['High'].rolling(window=10).max()
    df['Price_to_Support'] = (df['Price'] - df['Support_Level']) / df['Support_Level']
    df['Price_to_Resistance'] = (df['Resistance_Level'] - df['Price']) / df['Price']
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    return df

def train_all_stocks():
    """Train models for all stocks in the Train directory"""
    train_dir = 'Train'
    if not os.path.exists(train_dir):
        print("Train directory not found!")
        return
        
    # Get list of training files
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
    
    if not train_files:
        print("No training files found!")
        return
        
    print(f"Found {len(train_files)} stocks to train")
    
    # Train models for each stock
    for file in train_files:
        stock_name = file.split('.')[0]
        file_path = os.path.join(train_dir, file)
        train_and_save_models(stock_name, file_path)
        
    print("\nCompleted training all models!")

if __name__ == "__main__":
    train_all_stocks() 