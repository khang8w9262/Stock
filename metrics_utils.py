import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_percentage_error,
                           mean_squared_error,
                           mean_absolute_error,
                           r2_score)
import traceback

def calculate_prediction_metrics(df_predictions, filtered_data_for_calcs, end_date, max_date_in_data, show_success=None, prediction_type='normal'):
    """
    Calculate metrics for all prediction types and store them
    """
    global metrics_normal, metrics_future, metrics_supplement
    
    print("\nDEBUG: Starting metrics calculation...")
    print(f"DEBUG: Prediction type: {prediction_type}")
    print(f"DEBUG: Number of predictions: {len(df_predictions)}")

    def calculate_historical_metrics(y_true, y_pred):
        """Calculate metrics for historical predictions"""
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'R^2': r2_score(y_true, y_pred)
        }
        return metrics

    def calculate_future_metrics(predictions):
        """Calculate metrics for future predictions"""
        if len(predictions) < 2:
            return None
            
        # Calculate daily returns
        returns = np.diff(predictions) / predictions[:-1]
        
        # Trend analysis
        trend_direction = np.sum(returns > 0) / len(returns)
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Growth metrics
        total_return = (predictions[-1] - predictions[0]) / predictions[0]
        avg_daily_return = np.mean(returns)
        
        # Momentum
        recent_trend = np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)
        momentum = recent_trend / np.mean(returns) if np.mean(returns) != 0 else 0
        
        metrics = {
            'MSE': trend_direction,
            'MAE': volatility,
            'RMSE': abs(total_return),
            'MAPE': abs(avg_daily_return * 100),
            'R^2': momentum
        }
        return metrics

    def calculate_supplement_metrics(y_true, y_pred, baodautu_data=None):
        """Calculate metrics for supplementary predictions"""
        base_metrics = calculate_historical_metrics(y_true, y_pred)
        
        # Add supplementary metrics if baodautu data is available
        if baodautu_data is not None:
            # Calculate correlation with baodautu predictions
            common_dates = set(baodautu_data.index) & set(y_true.index)
            if common_dates:
                baodautu_pred = baodautu_data.loc[common_dates]
                correlation = np.corrcoef(y_pred[common_dates], baodautu_pred)[0, 1]
                base_metrics['Correlation_with_Baodautu'] = correlation
            
            # Calculate improvement over baodautu predictions
            if common_dates:
                baodautu_mse = mean_squared_error(y_true[common_dates], baodautu_pred)
                model_mse = mean_squared_error(y_true[common_dates], y_pred[common_dates])
                improvement = (baodautu_mse - model_mse) / baodautu_mse * 100
                base_metrics['Improvement_over_Baodautu'] = improvement
        
        return base_metrics

    try:
        metrics = {}
        
        # Process each model's predictions
        for model_name, df in df_predictions.items():
            if df.empty:
                continue
                
            if prediction_type == 'future':
                # Calculate future prediction metrics
                model_metrics = calculate_future_metrics(df['Prediction'].values)
                if model_metrics:
                    metrics[model_name] = model_metrics
                    
            elif prediction_type == 'supplement':
                # Get actual data for comparison
                actual_data = filtered_data_for_calcs[filtered_data_for_calcs['Date'].isin(df['Date'])]
                if not actual_data.empty and len(actual_data) == len(df):
                    y_true = actual_data.set_index('Date')['Price']
                    y_pred = df.set_index('Date')['Prediction']
                    
                    # Get baodautu data if available
                    baodautu_data = None
                    if 'Baodautu_Price' in df.columns:
                        baodautu_data = df.set_index('Date')['Baodautu_Price']
                    
                    metrics[model_name] = calculate_supplement_metrics(y_true, y_pred, baodautu_data)
                    
            else:  # Normal predictions
                actual_data = filtered_data_for_calcs[filtered_data_for_calcs['Date'].isin(df['Date'])]
                if not actual_data.empty and len(actual_data) == len(df):
                    y_true = actual_data['Price'].values
                    y_pred = df['Prediction'].values
                    metrics[model_name] = calculate_historical_metrics(y_true, y_pred)

        # Store metrics in appropriate variable based on prediction type
        if metrics:
            if prediction_type == 'future':
                metrics_future = metrics
            elif prediction_type == 'supplement':
                metrics_supplement = metrics
            else:
                metrics_normal = metrics
                
            if show_success:
                msg = "Đã cập nhật chỉ số cho "
                if prediction_type == 'future':
                    msg += "dự báo tương lai"
                elif prediction_type == 'supplement':
                    msg += "dự báo bổ sung"
                else:
                    msg += "dự báo thông thường"
                show_success(msg)
                
        return metrics

    except Exception as e:
        print(f"\nLỗi khi tính toán metrics: {str(e)}")
        traceback.print_exc()
        return None 