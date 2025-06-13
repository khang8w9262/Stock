import streamlit as st
import pandas as pd
import os
import sys
import json
from datetime import date, datetime
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.dates as mdates
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import traceback
import plotly.graph_objects as go

# Lấy đường dẫn tương đối của thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn các thư mục chính
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
VE_DIR = os.path.join(BASE_DIR, 'Ve')
BAODAUTU_DIR = os.path.join(BASE_DIR, 'Baodautu')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
CATBOOST_INFO_DIR = os.path.join(BASE_DIR, 'catboost_info')

# Đường dẫn file dữ liệu Báo Đầu Tư
BAODAUTU_ARTICLES = os.path.join(BAODAUTU_DIR, 'baodautu_articles.csv')
BAODAUTU_CHUYEN = os.path.join(BAODAUTU_DIR, 'baodautu_chuyen.csv')

def verify_directory(dir_path):
    """Kiểm tra và tạo thư mục, trả về thông tin chi tiết"""
    info = {
        'exists': False,
        'writable': False,
        'error': None,
        'files': [],
        'abs_path': os.path.abspath(dir_path)
    }
    
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            info['exists'] = True
        else:
            info['exists'] = True
        
        # Kiểm tra quyền ghi
        test_file = os.path.join(dir_path, 'test_write.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            info['writable'] = True
        except Exception as e:
            info['error'] = f"Không có quyền ghi: {str(e)}"
        
        # Liệt kê files
        if info['exists']:
            info['files'] = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            
    except Exception as e:
        info['error'] = str(e)
    
    return info

# Set page config FIRST
st.set_page_config(page_title="Dự báo chứng khoán Báo Đầu Tư", layout="wide")

# Kiểm tra thư mục khi khởi động
dirs_status = {}
for dir_name, dir_path in {
    'Train': TRAIN_DIR,
    'Ve': VE_DIR,
    'Baodautu': BAODAUTU_DIR,
    'models': MODELS_DIR,
    'metrics': METRICS_DIR
}.items():
    dirs_status[dir_name] = verify_directory(dir_path)

# Kiểm tra lỗi nghiêm trọng
critical_errors = []
if not dirs_status['Train']['exists'] or not dirs_status['Train']['writable']:
    critical_errors.append(f"Lỗi thư mục Train: {dirs_status['Train']['error']}")
if not dirs_status['Ve']['exists'] or not dirs_status['Ve']['writable']:
    critical_errors.append(f"Lỗi thư mục Ve: {dirs_status['Ve']['error']}")

if critical_errors:
    st.error("Lỗi nghiêm trọng:")
    for error in critical_errors:
        st.error(error)
    st.stop()

# Thêm đường dẫn để import các module backend
sys.path.append(os.path.abspath('.'))

# Import các hàm cần thiết
from cao import crawl_baodautu
import chuyen
from back import (
    get_train_files,
    load_data,
    load_model,
    load_scaler,
    load_feature_order,
    simulate_future_price,
    find_baodautu_prediction,
    retrain_all_models,
    create_advanced_features
)

# Import UI plotting functions
from UI import update_plot, update_plot_with_baodautu

def setup_directories():
    """Kiểm tra và tạo các thư mục cần thiết"""
    required_dirs = {
        'Baodautu': 'Thư mục chứa dữ liệu từ Báo Đầu Tư',
        'Train': 'Thư mục chứa dữ liệu huấn luyện',
        'Ve': 'Thư mục chứa dữ liệu để vẽ',
        'models': 'Thư mục chứa mô hình đã train',
        'metrics': 'Thư mục chứa metrics của hệ thống',
        'logs': 'Thư mục chứa logs',
        'catboost_info': 'Thư mục thông tin CatBoost'
    }
    
    status = {}
    for dir_name, description in required_dirs.items():
        dir_path = os.path.join(BASE_DIR, dir_name)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                status[dir_name] = f"Đã tạo thư mục {dir_name}"
            except Exception as e:
                status[dir_name] = f"Lỗi khi tạo thư mục {dir_name}: {str(e)}"
        else:
            status[dir_name] = f"Thư mục {dir_name} đã tồn tại"
            
        # Kiểm tra quyền ghi
        try:
            test_file = os.path.join(dir_path, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            status[dir_name] += " (có quyền ghi)"
        except Exception as e:
            status[dir_name] += f" (không có quyền ghi: {str(e)})"
    
    return status

def check_data_files():
    """Kiểm tra các file dữ liệu cần thiết"""
    status = {}
    
    # Kiểm tra file trong thư mục Train
    train_files = []
    if os.path.exists(TRAIN_DIR):
        train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv')]
    status['train_files'] = train_files
    
    # Kiểm tra file trong thư mục Ve
    ve_files = []
    if os.path.exists(VE_DIR):
        ve_files = [f for f in os.listdir(VE_DIR) if f.endswith('.csv')]
    status['ve_files'] = ve_files
    
    # Kiểm tra file Báo Đầu Tư
    status['baodautu_articles'] = os.path.exists(BAODAUTU_ARTICLES)
    status['baodautu_chuyen'] = os.path.exists(BAODAUTU_CHUYEN)
    
    return status

# Kiểm tra và tạo thư mục
dir_status = setup_directories()

# Kiểm tra file dữ liệu
data_status = check_data_files()

# Hiển thị thông tin hệ thống trong sidebar
with st.sidebar:
    # Menu chức năng
    st.markdown("### Menu chức năng")
    selected_function = st.radio(
        label="Chọn chức năng:",
        options=["1. Dự báo giá cổ phiếu", 
                "2. Phân tích sai số", 
                "3. Quản lý dữ liệu",
                "4. Cài đặt hệ thống"]
    )

# Tạo giao diện chính
st.title("Hệ thống dự báo chứng khoán Báo Đầu Tư")

def load_models(dataset):
    """Load all models for a given dataset"""
    models = {}
    model_types = ['rf', 'xgb', 'lgb', 'dt']
    
    print(f"Loading models for {dataset}...")
    for model_type in model_types:
        try:
            model = load_model(model_type, dataset)
            if model is not None:
                models[model_type] = model
                print(f"Loaded {model_type} model for {dataset}")
            else:
                print(f"Could not load {model_type} model for {dataset}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    return models

def get_predictions(model, df, dataset):
    """Get predictions from a model for the given data"""
    try:
        # Debug input data
        print("\n--- Getting Predictions ---")
        print("Input data shape:", df.shape)

        if df.empty:
            print("Input dataframe is empty. Cannot get predictions.")
            return None

        # Load scaler and feature order
        scaler = load_scaler(dataset)
        feature_order = load_feature_order(dataset)
        
        if scaler is None or feature_order is None:
            print(f"Could not load scaler or feature order for {dataset}")
            return None
            
        # Map column names to expected format
        df_mapped = df.copy()
        column_mapping = {
            'Lần cuối': 'Price',
            'Mở': 'Open', 
            'Cao': 'High',
            'Thấp': 'Low',
            'KL': 'Volume',
            '% Thay đổi': 'Change'
        }
        
        df_mapped.rename(columns=column_mapping, inplace=True)
        
        # Create features using mapped dataframe
        features_df = create_advanced_features(df_mapped)
        if features_df is None or features_df.empty:
            print("Could not create features.")
            return None
            
        # Ensure all required features are present
        missing_features = [f for f in feature_order if f not in features_df.columns]
        if missing_features:
            print(f"Error: Features missing from dataframe: {missing_features}")
            return None

        # Select and order features
        X = features_df[feature_order]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        print(f"Generated {len(predictions)} predictions.")
        return predictions
        
    except Exception as e:
        print(f"\nError in get_predictions: {e}")
        traceback.print_exc()
        return None

def plot_predictions(dataset, start_date=None, end_date=None, show_predictions=False, dark_theme=False, y_range=None):
    """Create interactive prediction plot"""
    try:
        print("\n" + "="*80)
        print(f"STARTING PLOT FOR: {dataset} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("="*80)

        # Build file paths and read full data
        file_path_train = os.path.join(TRAIN_DIR, f"{dataset}.csv")
        file_path_display = os.path.join(VE_DIR, f"{dataset}_TT.csv")
        
        df_train_full = pd.read_csv(file_path_train, encoding='utf-8', sep=',')
        df_display_full = pd.read_csv(file_path_display, encoding='utf-8', sep=',')
        
        # Convert dates and sort
        df_train_full['Date'] = pd.to_datetime(df_train_full['Ngày'], format='%d/%m/%Y')
        df_display_full['Date'] = pd.to_datetime(df_display_full['Ngày'], format='%d/%m/%Y')
        df_train_full = df_train_full.sort_values('Date').reset_index(drop=True)
        df_display_full = df_display_full.sort_values('Date').reset_index(drop=True)

        last_historical_date = df_train_full['Date'].max()

        fig = go.Figure()

        # Plot actual historical data within the selected range
        df_display_hist = df_display_full[
            (df_display_full['Date'] >= start_date) & (df_display_full['Date'] <= end_date)
        ]
        if not df_display_hist.empty:
            # Add a second, identical trace that is NOT in the legend, so it's always visible
            # and provides the hover info.
            fig.add_trace(go.Scatter(
                x=df_display_hist['Date'], y=df_display_hist['Lần cuối'],
                name='Giá thực tế', mode='lines', line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='x+y+name'
            ))
            # Add a trace for the legend that can be toggled, but has no hover info
            fig.add_trace(go.Scatter(
                x=df_display_hist['Date'], y=df_display_hist['Lần cuối'],
                name='Giá thực tế', mode='lines', line=dict(color='blue', width=2),
                hoverinfo='none'
            ))

        if show_predictions:
            models = load_models(dataset)
            colors = {'rf': 'red', 'xgb': 'orange', 'lgb': 'green', 'dt': 'purple'}
            dash_styles = {'rf': 'dash', 'xgb': 'dashdot', 'lgb': 'dot', 'dt': 'dash'}

            # --- Future Predictions ---
            if end_date > last_historical_date:
                print("\n--- Generating Future Forecast ---")
                forecast_start_date = max(start_date, last_historical_date + pd.Timedelta(days=1))
                future_dates = pd.bdate_range(start=forecast_start_date, end=end_date)

                if not future_dates.empty and models:
                    scaler = load_scaler(dataset)
                    
                    # Map column names for feature creation
                    df_train_full_mapped = df_train_full.copy()
                    df_train_full_mapped.rename(columns={
                        'Lần cuối': 'Price', 'Mở': 'Open', 'Cao': 'High',
                        'Thấp': 'Low', 'KL': 'Volume', '% Thay đổi': 'Change'
                    }, inplace=True)

                    all_features = create_advanced_features(df_train_full_mapped)
                    
                    if all_features is not None and not all_features.empty:
                        last_feature_vector = all_features.iloc[[-1]]
                        
                        for model_name, model in models.items():
                            print(f"Simulating future for {model_name.upper()}...")
                            future_predictions = simulate_future_price(
                                model, last_feature_vector, scaler, future_dates, stock_name=dataset
                            )
                            if future_predictions:
                                pred_dates, pred_values = zip(*future_predictions)
                                fig.add_trace(go.Scatter(
                                    x=list(pred_dates), y=list(pred_values),
                                    name=f'Dự báo (Tương lai) {model_name.upper()}',
                                    mode='lines', line=dict(color=colors.get(model_name, 'gray'), dash=dash_styles.get(model_name, 'solid'))
                                ))
                    else:
                        st.warning("Không thể tạo features để dự báo tương lai.")
            else:
                # --- Historical Predictions ---
                print("\n--- Generating Historical Predictions ---")
                
                # Filter the data to the selected range. We need df_train_full for feature creation.
                historical_prediction_data = df_train_full[
                    (df_train_full['Date'] >= start_date) & 
                    (df_train_full['Date'] <= end_date)
                ]

                if not historical_prediction_data.empty:
                    for model_name, model in models.items():
                        print(f"Generating historical predictions for {model_name.upper()}...")
                        
                        # We pass a copy to avoid SettingWithCopyWarning
                        predictions = get_predictions(model, historical_prediction_data.copy(), dataset)
                        
                        if predictions is not None and len(predictions) > 0:
                            # Align predictions with dates, as get_predictions might drop initial rows
                            prediction_dates = historical_prediction_data['Date'].iloc[-len(predictions):]
                            
                            fig.add_trace(go.Scatter(
                                x=prediction_dates, y=predictions,
                                name=f'Dự báo ({model_name.upper()})',
                                mode='lines', 
                                line=dict(color=colors.get(model_name, 'gray'), dash=dash_styles.get(model_name, 'solid'))
                            ))
                else:
                    print("No historical data in the selected range to generate predictions.")
        
        # Update layout
        fig.update_layout(
            title=dict(text=f'Biểu đồ giá {dataset}', font=dict(size=20)),
            xaxis=dict(title='Ngày', rangeslider=dict(visible=True), type='date'),
            yaxis=dict(title='Giá', tickformat='.2f'),
            hovermode='x unified',
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=1.05
            ),
            margin=dict(r=150),
            template='plotly_dark' if dark_theme else 'plotly_white'
        )
        fig.update_xaxes(
            range=[start_date, end_date],
            autorange=False,
            rangeslider_visible=True,
            rangeslider_range=[start_date, end_date],
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
        return fig

    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")
        traceback.print_exc()
        return None

def plot_predictions_with_baodautu(dataset, start_date, end_date, baodautu_predictions, dark_theme=False):
    """
    Create an interactive plot with ML predictions anchored to Baodautu data points.
    """
    try:
        print("\n" + "="*80)
        print(f"STARTING ANCHORED PLOT FOR: {dataset} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("="*80)

        # 1. Read historical data
        file_path_train = os.path.join(TRAIN_DIR, f"{dataset}.csv")
        file_path_display = os.path.join(VE_DIR, f"{dataset}_TT.csv")
        df_train_full = pd.read_csv(file_path_train, encoding='utf-8', sep=',')
        df_display_full = pd.read_csv(file_path_display, encoding='utf-8', sep=',')
        df_train_full['Date'] = pd.to_datetime(df_train_full['Ngày'], format='%d/%m/%Y')
        df_display_full['Date'] = pd.to_datetime(df_display_full['Ngày'], format='%d/%m/%Y')
        df_train_full = df_train_full.sort_values('Date').reset_index(drop=True)
        df_display_full = df_display_full.sort_values('Date').reset_index(drop=True)

        # 2. Start building the plot
        fig = go.Figure()

        # 3. Plot actual historical price
        df_display_hist = df_display_full[
            (df_display_full['Date'] >= start_date) & (df_display_full['Date'] <= end_date)
        ]
        if not df_display_hist.empty:
            # Add a second, identical trace that is NOT in the legend, so it's always visible
            # and provides the hover info.
            fig.add_trace(go.Scatter(
                x=df_display_hist['Date'], y=df_display_hist['Lần cuối'],
                name='Giá thực tế', mode='lines', line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='x+y+name'
            ))
            # Add a trace for the legend that can be toggled, but has no hover info
            fig.add_trace(go.Scatter(
                x=df_display_hist['Date'], y=df_display_hist['Lần cuối'],
                name='Giá thực tế', mode='lines', line=dict(color='blue', width=2),
                hoverinfo='none'
            ))

        # 4. Load models and support tools
        models = load_models(dataset)
        scaler = load_scaler(dataset)
        feature_order = load_feature_order(dataset)
        if not models or not scaler or not feature_order:
            st.error("Không thể tải models hoặc scaler, không thể tạo dự báo neo.")
            return None

        # 5. Prepare for simulation
        last_historical_date = df_train_full['Date'].max()
        last_price = df_train_full.iloc[-1]['Lần cuối']
        
        # Map column names for feature creation
        df_train_mapped = df_train_full.copy()
        df_train_mapped.rename(columns={'Lần cuối': 'Price', 'Mở': 'Open', 'Cao': 'High', 'Thấp': 'Low', 'KL': 'Volume', '% Thay đổi': 'Change'}, inplace=True)
        
        # Parse Volume column with 'K' and 'M' suffixes before calculating mean
        def parse_volume(value):
            if isinstance(value, (int, float)):
                return float(value)
            if not isinstance(value, str):
                return 0.0
            value = value.strip().upper().replace(',', '')
            if value.endswith('M'):
                try:
                    return float(value[:-1]) * 1_000_000
                except ValueError:
                    return 0.0
            elif value.endswith('K'):
                try:
                    return float(value[:-1]) * 1_000
                except ValueError:
                    return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        df_train_mapped['Volume'] = df_train_mapped['Volume'].apply(parse_volume).fillna(0)
        
        # Define anchor points: start from the last known price
        anchor_points = sorted([(last_historical_date, last_price)] + baodautu_predictions)
        colors = {'rf': 'red', 'xgb': 'orange', 'lgb': 'green', 'dt': 'purple'}
        dash_styles = {'rf': 'dash', 'xgb': 'dashdot', 'lgb': 'dot', 'dt': 'dash'}

        # 6. Main loop: for each model, generate the anchored prediction line
        for model_name, model in models.items():
            print(f"Simulating anchored predictions for {model_name.upper()}...")
            model_full_path = []

            # Loop through segments defined by anchors
            for i in range(len(anchor_points) - 1):
                start_anchor_date, start_anchor_price = anchor_points[i]
                end_anchor_date, end_anchor_price = anchor_points[i+1]
                
                segment_days = pd.bdate_range(start=start_anchor_date, end=end_anchor_date)
                if len(segment_days) < 2: continue

                # Create synthetic data for the segment
                t = np.linspace(0, 1, len(segment_days))
                price_curve = start_anchor_price + (end_anchor_price - start_anchor_price) * (3 * t**2 - 2 * t**3)
                
                segment_df = pd.DataFrame({'Date': segment_days, 'Price': price_curve})
                segment_df['Open'] = segment_df['Price'].shift(1).fillna(method='bfill')
                segment_df['High'] = segment_df[['Price', 'Open']].max(axis=1) * 1.01
                segment_df['Low'] = segment_df[['Price', 'Open']].min(axis=1) * 0.99
                segment_df['Volume'] = df_train_mapped['Volume'].mean()
                segment_df['Change'] = segment_df['Price'].pct_change().fillna(0)

                # Create features for this synthetic data
                combined_for_features = pd.concat([df_train_mapped.tail(30), segment_df], ignore_index=True)
                all_segment_features = create_advanced_features(combined_for_features)
                segment_features_only = all_segment_features[all_segment_features['Date'].isin(segment_days)]

                if not segment_features_only.empty:
                    X = segment_features_only[feature_order]
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    
                    # Ensure the line passes through anchors
                    predictions[0] = start_anchor_price
                    predictions[-1] = end_anchor_price
                    
                    model_full_path.extend(list(zip(segment_features_only['Date'], predictions)))
            
            # Plot the generated line for the model
            if model_full_path:
                dates, prices = zip(*model_full_path)
                fig.add_trace(go.Scatter(
                    x=dates, y=prices,
                    name=f'Dự báo ({model_name.upper()})',
                    mode='lines', line=dict(color=colors.get(model_name, 'gray'), dash=dash_styles.get(model_name, 'solid'))
                ))

        # 7. Plot the anchor markers
        bd_dates, bd_prices = zip(*baodautu_predictions)
        fig.add_trace(go.Scatter(
            x=bd_dates, y=bd_prices, name='Dự báo Báo Đầu Tư', mode='markers',
            marker=dict(color='cyan', size=10, symbol='star')
        ))
        
        # 8. Update layout
        fig.update_layout(
            title=dict(text=f'Biểu đồ giá {dataset} (với Báo Đầu Tư)', font=dict(size=20)),
            xaxis=dict(
                title='Ngày', 
                rangeslider=dict(visible=True, range=[start_date, end_date]), 
                type='date', 
                range=[start_date, end_date], 
                autorange=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            ),
            yaxis=dict(title='Giá', tickformat='.2f'),
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
            margin=dict(r=150),
            template='plotly_dark' if dark_theme else 'plotly_white'
        )
        return fig

    except Exception as e:
        print(f"Error in plot_predictions_with_baodautu: {str(e)}")
        traceback.print_exc()
        return None

# Hiển thị nội dung tương ứng với chức năng được chọn
if selected_function == "1. Dự báo giá cổ phiếu":
    # Get list of stocks
    try:
        if not data_status['train_files']:
            st.error("Không tìm thấy dữ liệu huấn luyện trong thư mục Train!")
            st.error("Vui lòng thêm file CSV vào thư mục Train trước khi sử dụng.")
            st.stop()
            
        stock_files = [os.path.join(TRAIN_DIR, f) for f in data_status['train_files']]
        stock_names = [os.path.splitext(os.path.basename(f))[0] for f in stock_files]
        stock_names.sort()
        
        # Kiểm tra file Ve tương ứng
        missing_ve_files = []
        for stock in stock_names:
            ve_file = f"{stock}_TT.csv"
            if ve_file not in data_status['ve_files']:
                missing_ve_files.append(ve_file)
        
        if missing_ve_files:
            st.warning("Một số file dữ liệu vẽ đồ thị bị thiếu:")
            for f in missing_ve_files:
                st.warning(f"- {f}")
            
    except Exception as e:
        st.error(f"Không thể tải danh sách cổ phiếu: {e}")
        st.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        st.stop()

    if not stock_names:
        st.warning("Không tìm thấy dữ liệu cổ phiếu đã huấn luyện trong thư mục 'Train'.")
    else:
        try:
            # Debug info
            st.sidebar.markdown("### Debug Information")
            debug_container = st.sidebar.expander("Show Debug Info", expanded=False)
            
            # Tạo container cho phần controls
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
                
                with col1:
                    selected_stock = st.selectbox("Chọn mã chứng khoán", stock_names)
                
                with col2:
                    start_date = st.date_input("Ngày bắt đầu", 
                                             value=date.today(),
                                             format="DD/MM/YYYY")
                
                with col3:
                    end_date = st.date_input("Ngày kết thúc",
                                           value=date.today() + pd.Timedelta(days=30),
                                           format="DD/MM/YYYY")
                
                with col4:
                    st.write("")
                    st.write("")
                    if st.button("Huấn luyện lại"):
                        with st.spinner("Đang huấn luyện lại các mô hình..."):
                            retrain_all_models()
                        st.success("Đã huấn luyện lại tất cả mô hình!")
                        st.cache_data.clear()

            # Add buttons for additional functionality
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                forecast_button = st.button("Cập nhật dự báo")
            with col_btn2:
                supplement_button = st.button("Bổ sung Báo Đầu Tư")

            # Tạo container cho metrics
            metrics_container = st.container()

            # Tạo container cho biểu đồ
            chart_container = st.container()
            
            # Debug info
            with debug_container:
                st.write("Selected Stock:", selected_stock)
                st.write("Start Date:", start_date)
                st.write("End Date:", end_date)
                st.write("File Paths:")
                st.write(f"- Train: {os.path.join(TRAIN_DIR, f'{selected_stock}.csv')}")
                st.write(f"- Display: {os.path.join(VE_DIR, f'{selected_stock}_TT.csv')}")
            
            # Hiển thị biểu đồ
            if start_date <= end_date:
                try:
                    start_date_pd = pd.to_datetime(start_date)
                    end_date_pd = pd.to_datetime(end_date)
                    
                    with debug_container:
                        st.write("Converted Dates:")
                        st.write(f"- Start: {start_date_pd}")
                        st.write(f"- End: {end_date_pd}")
                    
                    fig = None
                    # Decide which plot to generate based on the button clicked
                    if supplement_button:
                        with st.spinner(f"Đang tạo dự báo neo theo Báo Đầu Tư cho {selected_stock}..."):
                            baodautu_predictions = []
                            current_date = start_date_pd
                            while current_date <= end_date_pd:
                                price, found = find_baodautu_prediction(selected_stock, current_date)
                                if found:
                                    baodautu_predictions.append((current_date, price))
                                current_date += pd.Timedelta(days=1)
                            
                            if not baodautu_predictions:
                                st.warning(f"Không tìm thấy dự báo Báo Đầu Tư cho {selected_stock}. Hiển thị dự báo thông thường.")
                                fig = plot_predictions(selected_stock, start_date_pd, end_date_pd, show_predictions=True, dark_theme=True)
                            else:
                                fig = plot_predictions_with_baodautu(selected_stock, start_date_pd, end_date_pd, baodautu_predictions, dark_theme=True)
                    else:
                         with st.spinner("Đang tạo biểu đồ dự báo..."):
                            fig = plot_predictions(selected_stock, start_date_pd, end_date_pd, show_predictions=True, dark_theme=True)

                    if fig is not None:
                        with debug_container:
                            st.write("Figure created successfully")
                            
                        with chart_container:
                            st.markdown("### Biểu đồ dự báo chi tiết")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Không thể tạo biểu đồ. Vui lòng kiểm tra debug info để biết thêm chi tiết.")
                
                except Exception as e:
                    st.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
                    st.error(f"Chi tiết lỗi: {traceback.format_exc()}")
            else:
                st.error("Ngày kết thúc phải sau ngày bắt đầu.")

        except Exception as e:
            st.error(f"Lỗi không xác định: {str(e)}")
            st.error(f"Chi tiết lỗi: {traceback.format_exc()}")

elif selected_function == "2. Phân tích sai số":
    metrics_file = os.path.join('metrics', 'system_metrics.json')
    
    if not os.path.exists(metrics_file):
        st.warning("Chưa có dữ liệu sai số. Hãy chạy dự báo trước.")
    else:
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            metrics_list = []
            for pred_type, models_data in metrics_data.items():
                if isinstance(models_data, dict):
                    for model_name, scores in models_data.items():
                        if isinstance(scores, dict):
                            row = {'Loại dự báo': pred_type, 'Mô hình': model_name.upper()}
                            row.update(scores)
                            metrics_list.append(row)
            
            if not metrics_list:
                st.info("File metrics rỗng hoặc có định dạng không đúng.")
                st.json(metrics_data)
            else:
                df_metrics = pd.DataFrame(metrics_list)
                
                cols_order = ['Loại dự báo', 'Mô hình', 'MSE', 'MAE', 'RMSE', 'MAPE', 'R^2']
                existing_cols = [c for c in cols_order if c in df_metrics.columns]
                df_metrics = df_metrics[existing_cols]

                # Format a copy of the dataframe to avoid warning
                df_display = df_metrics.copy()
                format_dict = {
                    'MSE': '{:.6f}', 'MAE': '{:.6f}', 'RMSE': '{:.6f}',
                    'MAPE': '{:.6f}', 'R^2': '{:.6f}'
                }
                for col, fmt in format_dict.items():
                    if col in df_display.columns:
                        df_display[col] = df_display[col].apply(lambda x: fmt.format(x) if isinstance(x, (int, float)) else x)
                
                st.dataframe(df_display, use_container_width=True)
        except Exception as e:
            st.error(f"Không thể đọc hoặc xử lý file metrics: {e}")

elif selected_function == "3. Quản lý dữ liệu":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cào dữ liệu Báo Đầu Tư"):
            with st.spinner("Đang cào dữ liệu..."):
                try:
                    crawl_baodautu(num_pages=2, output_csv=BAODAUTU_ARTICLES)
                    st.success("Đã cào xong dữ liệu!")
                except Exception as e:
                    st.error(f"Lỗi khi cào dữ liệu: {str(e)}")
    with col2:
        if st.button("Chuyển đổi dữ liệu Báo Đầu Tư"):
            with st.spinner("Đang chuyển đổi dữ liệu..."):
                try:
                    # Đọc file articles
                    if os.path.exists(BAODAUTU_ARTICLES):
                        df = pd.read_csv(BAODAUTU_ARTICLES)
                        # Phân tích tất cả bài viết và lưu ra file chuyển
                        chuyen.phan_tich_tat_ca_bai_viet(df, BAODAUTU_CHUYEN)
                        st.success("Đã chuyển đổi và lưu dữ liệu!")
                    else:
                        st.error(f"Không tìm thấy file {BAODAUTU_ARTICLES}. Đường dẫn hiện tại: {os.path.abspath(BAODAUTU_ARTICLES)}")
                except Exception as e:
                    st.error(f"Lỗi khi chuyển đổi dữ liệu: {str(e)}")
    st.subheader("Bảng dữ liệu baodautu_articles.csv")
    try:
        if os.path.exists(BAODAUTU_ARTICLES):
            df1 = pd.read_csv(BAODAUTU_ARTICLES, on_bad_lines='skip')
            st.dataframe(df1)
        else:
            st.warning("Chưa có dữ liệu baodautu_articles.csv")
    except Exception as e:
        st.error(f"Lỗi khi đọc baodautu_articles.csv: {e}")
    st.subheader("Bảng dữ liệu baodautu_chuyen.csv")
    try:
        if os.path.exists(BAODAUTU_CHUYEN):
            df2 = pd.read_csv(BAODAUTU_CHUYEN, on_bad_lines='skip')
            st.dataframe(df2)
        else:
            st.warning("Chưa có dữ liệu baodautu_chuyen.csv")
    except Exception as e:
        st.error(f"Lỗi khi đọc baodautu_chuyen.csv: {e}")

elif selected_function == "4. Cài đặt hệ thống":
    st.write("Phần cài đặt hệ thống") 