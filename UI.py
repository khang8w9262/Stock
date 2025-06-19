import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import traceback
import json

# Đường dẫn tuyệt đối
BASE_DIR = 'D:\NghienCuu\Stock'

# Đường dẫn các thư mục
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
VE_DIR = os.path.join(BASE_DIR, 'Ve')
BAODAUTU_DIR = os.path.join(BASE_DIR, 'Baodautu')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
CATBOOST_INFO_DIR = os.path.join(BASE_DIR, 'catboost_info')

# Đường dẫn file dữ liệu
BAODAUTU_ARTICLES = os.path.join(BAODAUTU_DIR, 'baodautu_articles.csv')
BAODAUTU_CHUYEN = os.path.join(BAODAUTU_DIR, 'baodautu_chuyen.csv')

# Import backend functions from back.py
from back import (load_data, create_advanced_features, simulate_future_price, 
                 build_lightgbm_model, load_model, save_model, load_scaler, save_scaler,
                 find_baodautu_prediction, load_feature_order)
from metrics_utils import calculate_prediction_metrics

# Load danh sách mã cổ phiếu
if os.path.exists(TRAIN_DIR):
    stock_files = [f.split('.')[0] for f in os.listdir(TRAIN_DIR) if f.endswith('.csv')]
    if not stock_files:
        stock_files = ["AAPL", "MSFT", "GOOGL"]
else:
    stock_files = ["AAPL", "MSFT", "GOOGL"]
stock_files.sort()

# Sidebar chọn mã cổ phiếu
st.sidebar.title("Chọn mã cổ phiếu")
stock = st.sidebar.selectbox("Mã cổ phiếu", stock_files)

# Nhập ngày bắt đầu và kết thúc
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Ngày bắt đầu", value=datetime(2023, 1, 1))
with col2:
    end_date = st.date_input("Ngày kết thúc", value=datetime(2023, 12, 31))

# Nút dự báo
if st.button("Dự báo"):
    try:
        file_path_train = os.path.join(TRAIN_DIR, f"{stock}.csv")
        file_path_display = os.path.join(VE_DIR, f"{stock}_TT.csv")
        if not os.path.exists(file_path_train) or not os.path.exists(file_path_display):
            st.error(f"Không tìm thấy dữ liệu cho {stock}!")
        else:
            df_train = pd.read_csv(file_path_train, encoding='utf-8', sep=',')
            df_display = pd.read_csv(file_path_display, encoding='utf-8', sep=',')
            df_display['Date'] = pd.to_datetime(df_display['Ngày'], format='%d/%m/%Y')
            mask = (df_display['Date'] >= pd.to_datetime(start_date)) & (df_display['Date'] <= pd.to_datetime(end_date))
            df_display = df_display[mask]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_display['Date'], df_display['Lần cuối'], label='Giá thực tế', color='blue', linewidth=2)
            ax.set_title(f'Biểu đồ giá {stock}', fontsize=14)
            ax.set_xlabel('Ngày', fontsize=12)
            ax.set_ylabel('Giá', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Lỗi khi cập nhật biểu đồ: {e}")
        st.text(traceback.format_exc())

# Hiển thị bảng metrics
if st.button("Xem metrics"):
    metrics_file = os.path.join('metrics', 'system_metrics.json')
    if not os.path.exists(metrics_file):
        st.warning("Không tìm thấy dữ liệu metrics. Vui lòng chạy tính toán metrics trước.")
    else:
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            for pred_type, title in {'normal': 'Dự báo thông thường', 'future': 'Dự báo tương lai', 'supplement': 'Dự báo bổ sung'}.items():
                if pred_type in metrics_data and metrics_data[pred_type]:
                    st.subheader(title)
                    df = pd.DataFrame(metrics_data[pred_type]).T
                    st.dataframe(df)
        except Exception as e:
            st.error(f"Lỗi khi đọc metrics: {e}")

# Hiển thị bảng dữ liệu
if st.checkbox("Xem dữ liệu gốc"):
    file_path_display = os.path.join(VE_DIR, f"{stock}_TT.csv")
    if os.path.exists(file_path_display):
        df_display = pd.read_csv(file_path_display, encoding='utf-8', sep=',')
        st.dataframe(df_display)
    else:
        st.warning("Không tìm thấy file dữ liệu hiển thị.")

# Gợi ý: Bạn có thể bổ sung thêm các nút, bảng, hoặc các chức năng dự báo nâng cao khác bằng các widget của Streamlit.
