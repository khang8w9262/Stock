import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error,
                             mean_absolute_error,
                             r2_score)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
import hashlib
import joblib

# Đường dẫn tuyệt đối
BASE_DIR = 'D:\\NghienCuu\\Stock'

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

# Kiểm tra và tạo các thư mục nếu chưa tồn tại
required_dirs = [
    TRAIN_DIR,
    VE_DIR,
    BAODAUTU_DIR,
    MODELS_DIR,
    METRICS_DIR,
    LOGS_DIR,
    CATBOOST_INFO_DIR
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"Đã tạo thư mục: {dir_path}")
        except Exception as e:
            print(f"Không thể tạo thư mục {dir_path}: {e}")

def validate_models_dir():
    """
    Kiểm tra và đảm bảo thư mục models tồn tại và có quyền ghi
    """
    global models_dir
    try:
        # Thử tạo một file tạm để kiểm tra quyền ghi
        test_file = os.path.join(models_dir, 'test.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except Exception as e:
        print(f"Lỗi khi kiểm tra quyền ghi thư mục models: {e}")
        # Thử chuyển sang thư mục hiện tại
        models_dir = 'models'
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir)
            except Exception as e2:
                print(f"Không thể tạo thư mục models tại thư mục hiện tại: {e2}")
                return False
        return validate_models_dir()

# Kiểm tra thư mục models khi khởi động
validate_models_dir()

def save_model(model, model_name, stock_name):
    """
    Lưu model vào thư mục models với tên duy nhất cho mỗi cổ phiếu
    """
    try:
        # Kiểm tra lại thư mục models
        if not validate_models_dir():
            print("Không thể truy cập thư mục models")
            return False
        
        # Tạo tên file model theo format: STOCK_model_name_model.joblib
        filename = f"{stock_name}_{model_name}_model.joblib"
        filepath = os.path.join(models_dir, filename)
        
        # Debug thông tin về đường dẫn
        print(f"Đang lưu model tại: {os.path.abspath(filepath)}")
        print(f"Thư mục models hiện tại: {os.path.abspath(models_dir)}")
        
        # Lưu model
        joblib.dump(model, filepath)
        print(f"Đã lưu model {model_name} cho {stock_name} tại {filepath}")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu model {model_name} cho {stock_name}: {e}")
        return False

def load_model(model_name, stock_name):
    """
    Tải model từ thư mục models
    """
    try:
        # Kiểm tra lại thư mục models
        if not validate_models_dir():
            print("Không thể truy cập thư mục models")
            return None
            
        # Tạo tên file model theo format: STOCK_model_name_model.joblib
        filename = f"{stock_name}_{model_name}_model.joblib"
        filepath = os.path.join(models_dir, filename)
        
        # Debug thông tin về đường dẫn
        print(f"Đang tìm model tại: {os.path.abspath(filepath)}")
        print(f"Thư mục models hiện tại: {os.path.abspath(models_dir)}")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(filepath):
            print(f"Không tìm thấy model {model_name} cho {stock_name}")
            # Kiểm tra thư mục models
            if os.path.exists(models_dir):
                print(f"Các file trong thư mục {models_dir}:")
                for file in os.listdir(models_dir):
                    print(f"  - {file}")
            else:
                print(f"Thư mục {models_dir} không tồn tại")
            return None
            
        # Tải model
        try:
            model = joblib.load(filepath)
            print(f"Đã tải model {model_name} cho {stock_name} từ {filepath}")
            return model
        except Exception as e:
            print(f"Lỗi khi tải model từ file {filepath}: {e}")
            return None
            
    except Exception as e:
        print(f"Lỗi khi tải model {model_name} cho {stock_name}: {e}")
        return None

def save_scaler(scaler, stock_name):
    """
    Lưu scaler vào thư mục models
    """
    try:
        # Tạo tên file scaler theo format: STOCK_scaler.joblib
        filename = f"{stock_name}_scaler.joblib"
        filepath = os.path.join(models_dir, filename)
        
        # Lưu scaler
        joblib.dump(scaler, filepath)
        print(f"Đã lưu scaler cho {stock_name} tại {filepath}")
        return True
    except Exception as e:
        print(f"Lỗi khi lưu scaler cho {stock_name}: {e}")
        return False

def load_scaler(stock_name):
    """
    Tải scaler từ thư mục models
    """
    try:
        # Tạo tên file scaler theo format: STOCK_scaler.joblib
        filename = f"{stock_name}_scaler.joblib"
        filepath = os.path.join(models_dir, filename)
        
        # Kiểm tra file tồn tại
        if not os.path.exists(filepath):
            print(f"Không tìm thấy scaler cho {stock_name}")
            return None
            
        # Tải scaler
        scaler = joblib.load(filepath)
        print(f"Đã tải scaler cho {stock_name} từ {filepath}")
        return scaler
    except Exception as e:
        print(f"Lỗi khi tải scaler cho {stock_name}: {e}")
        return None

# Check if TensorFlow is available
try:
    # Libraries for LSTM (Keras/TensorFlow)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras import backend as K
    import tensorflow as tf
    # Turn off unnecessary TF warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False

warnings.filterwarnings("ignore")

# ---------------------------
# Data processing functions
# ---------------------------

# Create sample CSV file when needed
def create_sample_csv(file_path, sample_type='stock'):
    """
    Create a sample CSV file for stocks or drawing files
    """
    try:
        print(f"Creating sample CSV file at: {file_path}")
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Create sample data for stocks
        import datetime
        import random
        
        today = datetime.datetime.now()
        dates = [(today - datetime.timedelta(days=i)).strftime('%d/%m/%Y') for i in range(100, 0, -1)]
        
        base_price = 1000.0 if sample_type == 'stock' else 1000.0
        price = base_price
        
        data = []
        for date in dates:
            change_pct = random.uniform(-0.05, 0.05)
            price = price * (1 + change_pct)
            price = max(price, 100)  # Ensure price is not lower than 100
            
            # Calculate other values
            open_price = price * random.uniform(0.95, 1.05)
            high_price = max(price, open_price) * random.uniform(1.0, 1.1)
            low_price = min(price, open_price) * random.uniform(0.9, 1.0)
            volume = int(random.uniform(10000, 1000000))
            change = change_pct * 100  # Percentage change
            
            data.append([date, round(price, 2), round(open_price, 2), 
                        round(high_price, 2), round(low_price, 2), 
                        volume, round(change, 2)])
        
        # Create DataFrame and write to CSV file
        import pandas as pd
        df = pd.DataFrame(data, columns=['Ngày', 'Lần cuối', 'Mở', 'Cao', 'Thấp', 'KL', '% Thay đổi'])
        df.to_csv(file_path, index=False)
        
        print(f"Sample CSV file created successfully: {file_path}")
        return True
    except Exception as e:
        print(f"Error creating sample CSV file: {e}")
        return False

def load_data(file_path):
    """
    Read data from CSV file with different formats
    """
    print(f"Reading file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return pd.DataFrame()
    
    # Try with different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1', 'utf-16']
    
    # Try with different separators
    separators = [',', ';', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                print(f"Trying to read file with encoding: {encoding}, separator: {sep}")
                
                # Read CSV file with flexible parameters and handle bad lines
                data = pd.read_csv(file_path, encoding=encoding, sep=sep, 
                                 on_bad_lines='skip')
                
                # Check the data read
                if data.empty:
                    print(f"No data when reading with encoding {encoding}, separator {sep}")
                    continue
                    
                print(f"Successfully read file with encoding {encoding}, separator {sep}")
                print(f"Number of columns: {len(data.columns)}")
                print(f"Number of rows: {len(data)}")
                print(f"Columns: {data.columns.tolist()}")
                print(f"Sample data (first 3 rows):")
                print(data.head(3))
                
                # If only 1 column, the file might be read with wrong separator
                if len(data.columns) == 1:
                    print("Only 1 column. Might be wrong separator. Try another separator.")
                    continue
                
                # Ensure enough columns
                if len(data.columns) < 6:
                    print(f"Warning: File doesn't have enough columns. Current columns: {len(data.columns)}")
                    
                    # If has exactly the right number of columns but all in one, might need to split
                    if len(data.columns) == 1 and ',' in str(data.iloc[0, 0]):
                        print("Trying to split first column")
                        first_column = data.columns[0]
                        try:
                            # Try to split the first column by comma
                            split_data = data[first_column].str.split(',', expand=True)
                            if len(split_data.columns) >= 6:
                                data = split_data
                                print(f"Successfully split into {len(split_data.columns)} columns")
                            else:
                                continue
                        except:
                            continue
                    else:
                        # Not enough columns, try another separator
                        continue
                
                # Define required columns
                required_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume', 'Change']
                
                # If exactly 7 columns and column names are not required_cols
                if len(data.columns) == 7 and list(data.columns) != required_cols:
                    # Rename columns in order
                    data.columns = required_cols
                # If column names can be recognized from Vietnamese or English
                elif len(data.columns) >= 7:
                    # Map column names
                    column_map = {}
                    
                    # Keywords for each type of column
                    date_keywords = ['ngay', 'ngày', 'date', 'thoi gian', 'thời gian', 'time']
                    price_keywords = ['gia', 'giá', 'price', 'close', 'đóng', 'dong', 'last', 'cuoi', 'cuối', 'lần cuối', 'lan cuoi']
                    open_keywords = ['open', 'mo', 'mở', 'mo cua', 'mở cửa']
                    high_keywords = ['high', 'cao', 'cao nhat', 'cao nhất', 'max']
                    low_keywords = ['low', 'thap', 'thấp', 'thap nhat', 'thấp nhất', 'min']
                    volume_keywords = ['volume', 'khoi luong', 'khối lượng', 'kl', 'vol', 'klgd']
                    change_keywords = ['change', 'thay doi', 'thay đổi', '%', 'pct', 'percent']
                    
                    # Search for each type of column
                    for col in data.columns:
                        col_lower = str(col).lower()
                        if any(keyword in col_lower for keyword in date_keywords):
                            column_map[col] = 'Date'
                        elif any(keyword in col_lower for keyword in price_keywords):
                            column_map[col] = 'Price'
                        elif any(keyword in col_lower for keyword in open_keywords):
                            column_map[col] = 'Open'
                        elif any(keyword in col_lower for keyword in high_keywords):
                            column_map[col] = 'High'
                        elif any(keyword in col_lower for keyword in low_keywords):
                            column_map[col] = 'Low'
                        elif any(keyword in col_lower for keyword in volume_keywords):
                            column_map[col] = 'Volume'
                        elif any(keyword in col_lower for keyword in change_keywords):
                            column_map[col] = 'Change'
                    
                    # Rename identified columns
                    for old_col, new_col in column_map.items():
                        data = data.rename(columns={old_col: new_col})
                    
                    # Check if we have the required columns
                    missing_cols = [col for col in ['Date', 'Price'] if col not in data.columns]
                    if missing_cols:
                        print(f"Missing important columns: {missing_cols}")
                        continue
                    
                    # If missing other columns, use Price to fill
                    for col in ['Open', 'High', 'Low']:
                        if col not in data.columns:
                            data[col] = data['Price']
                    
                    # If missing Volume or Change, add with default values
                    if 'Volume' not in data.columns:
                        data['Volume'] = 0
                    if 'Change' not in data.columns:
                        data['Change'] = 0
                else:
                    # Columns do not match
                    print(f"Cannot identify required columns. Number of columns: {len(data.columns)}")
                    continue
                
                # Convert Date column
                try:
                    # Try different date formats
                    date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']
                    
                    for date_format in date_formats:
                        try:
                            data['Date'] = pd.to_datetime(data['Date'], format=date_format, errors='coerce')
                            # If less than 50% are NaT, this format might be correct
                            if data['Date'].isna().mean() < 0.5:
                                print(f"Successfully converted dates with format: {date_format}")
                                break
                        except:
                            continue
                    
                    # If still not successful, try automatic conversion
                    if data['Date'].isna().mean() >= 0.5:
                        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                    
                    # Check for NaT values
                    nat_count = data['Date'].isna().sum()
                    if nat_count > 0:
                        print(f"There are {nat_count} date values that could not be converted")
                except Exception as e:
                    print(f"Error converting Date column: {e}")
                    continue
                
                # Convert numeric columns
                for col in ['Price', 'Open', 'High', 'Low', 'Volume', 'Change']:
                    try:
                        # Convert directly to numbers, skip sample checking
                        if col == 'Volume':
                            # Special handling for Volume column
                            data[col] = data[col].astype(str).str.replace(',', '')
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            data[col] = data[col].fillna(0)
                        elif col == 'Change':
                            # Special handling for Change column (percentage)
                            data[col] = data[col].astype(str).str.replace('%', '').str.replace(',', '.')
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            data[col] = data[col].fillna(0)
                        else:
                            # Handling for price columns
                            data[col] = data[col].astype(str).str.replace(',', '')
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                            # Use forward fill and backward fill for missing values
                            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                            
                    except Exception as e:
                        print(f"Error processing column {col}: {e}")
                        if col in ['Volume', 'Change']:
                            data[col] = 0
                        else:
                            # For price columns, take the nearest value
                            data[col] = data['Price'] if 'Price' in data else 0
                
                # Only filter rows with Date and Price
                original_rows = len(data)
                data = data.dropna(subset=['Date', 'Price'])
                rows_dropped = original_rows - len(data)
                if rows_dropped > 0:
                    print(f"Dropped {rows_dropped} rows without date or price")
                
                # Sort by Date and clean index
                data = data.sort_values('Date')
                data = data.reset_index(drop=True)
                
                # Check final data
                if not data.empty:
                    print(f"Finished processing data for {os.path.basename(file_path)}:")
                    print(f"- Rows: {len(data)}")
                    print(f"- Date range: {data['Date'].min().strftime('%d/%m/%Y')} to {data['Date'].max().strftime('%d/%m/%Y')}")
                else:
                    print(f"Warning: No data after processing file {os.path.basename(file_path)}")
                
                return data
            except Exception as e:
                print(f"Error reading with encoding {encoding}, separator {sep}: {e}")
                continue
    
    # If we've tried all ways and haven't succeeded
    print(f"Could not read file {file_path} with any encoding and separator")
    
    # Try reading as Excel file
    if file_path.endswith('.csv'):
        try:
            print("Trying to read as Excel file...")
            data = pd.read_excel(file_path, engine='openpyxl')
            if not data.empty:
                print("Successfully read as Excel file")
                print(f"Number of columns: {len(data.columns)}")
                print(f"Number of rows: {len(data)}")
                # Continue processing as with CSV...
                return data
        except Exception as e:
            print(f"Could not read as Excel file: {e}")
    
    return pd.DataFrame()

# Add new function to handle direct number input
def parse_european_number(value_str):
    """
    Parse a European format number string (1.266,78) into a float
    """
    if not isinstance(value_str, str):
        return value_str
        
    # Check if it's a European format number (dots for thousands, comma for decimal)
    if '.' in value_str and ',' in value_str and value_str.find('.') < value_str.find(','):
        # Remove dots (thousands separator)
        value_str = value_str.replace('.', '')
        # Replace comma with dot (decimal separator)
        value_str = value_str.replace(',', '.')
        
    # Try to convert to number
    try:
        return float(value_str)
    except:
        # If error, return original value
        return value_str

def save_feature_order(feature_names, stock_name):
    """
    Save feature names and their order used during training
    """
    try:
        filename = f"{stock_name}_feature_order.joblib"
        filepath = os.path.join(models_dir, filename)
        joblib.dump(feature_names, filepath)
        print(f"Saved feature order for {stock_name}")
        return True
    except Exception as e:
        print(f"Error saving feature order for {stock_name}: {e}")
        return False

def load_feature_order(stock_name):
    """
    Load feature names and their order used during training
    """
    try:
        filename = f"{stock_name}_feature_order.joblib"
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            print(f"Feature order file not found for {stock_name}")
            return None
        feature_names = joblib.load(filepath)
        print(f"Loaded feature order for {stock_name}")
        return feature_names
    except Exception as e:
        print(f"Error loading feature order for {stock_name}: {e}")
        return None

def simulate_future_price(model, last_features_df, scaler, future_dates, historical_volatility=0.01, stock_name=None):
    """
    Simulate stock prices in the future using an iterative approach.
    """
    predictions = []
    
    # Load the feature order used during training
    feature_order = load_feature_order(stock_name)
    if feature_order is None:
        print(f"Could not load feature order for {stock_name}")
        return predictions
    
    # Check for empty inputs
    if future_dates.empty or last_features_df.empty:
        print("Cannot simulate future price: future_dates or last_features_df is empty.")
        return predictions

    # Start with the last known set of features
    current_features = last_features_df.copy().reset_index(drop=True)

    # Use the last known price as the base for the simulation
    last_price = current_features['Price'].iloc[0]
    
    print(f"Starting future simulation with base price: {last_price:.2f}")

    # Create realistic-looking random noise for the simulation period
    np.random.seed(42)
    random_noise = np.random.normal(0, historical_volatility, len(future_dates))
    
    for i, date in enumerate(future_dates):
        try:
            # Update time-based features for the current prediction day
            current_features.loc[0, 'DayOfWeek'] = float(date.dayofweek)
            current_features.loc[0, 'DayOfWeek_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
            current_features.loc[0, 'DayOfWeek_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)

            # Ensure features are in the correct order for prediction
            features_for_prediction = current_features[feature_order]
            
            # Scale features and make a prediction
            features_scaled = scaler.transform(features_for_prediction)
            predicted_price = float(model.predict(features_scaled)[0])
            
            # Add some controlled randomness to make the forecast less linear
            adjusted_price = predicted_price * (1 + random_noise[i])
            
            # Apply smoothing to prevent wild jumps
            smoothing_factor = 0.7
            adjusted_price = (smoothing_factor * adjusted_price) + ((1 - smoothing_factor) * last_price)
            
            # Store the prediction for this date
            predictions.append((date, adjusted_price))
            
            # --- Update features for the next iteration ---
            # This part is simplified. A more complex model would recalculate all features.
            
            # Shift lag features: lag_2 gets lag_1's value, etc.
            for j in range(5, 1, -1):
                current_features.loc[0, f'lag_{j}'] = current_features.loc[0, f'lag_{j-1}']
            current_features.loc[0, 'lag_1'] = last_price

            # Update the price to the newly predicted price
            current_features.loc[0, 'Price'] = adjusted_price
            
            # Update other simple features based on the new price
            # For simplicity, we keep Open, High, Low relative to the new price
            current_features.loc[0, 'Open'] = last_price # Today's open is yesterday's close
            current_features.loc[0, 'High'] = max(adjusted_price, last_price)
            current_features.loc[0, 'Low'] = min(adjusted_price, last_price)
            current_features.loc[0, 'Change'] = (adjusted_price - last_price) / last_price if last_price else 0
            
            # Update the 'last_price' for the next loop
            last_price = adjusted_price
            
        except Exception as e:
            print(f"Error during simulation on {date.strftime('%Y-%m-%d')}: {e}")
            # If an error occurs, just carry over the last price to avoid crashing
            if last_price:
                predictions.append((date, last_price))
    
    return predictions

def create_advanced_features(data, n_lags=5):
    """
    Create advanced features for stock data
    """
    try:
        df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume', 'Change']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}")
            # Try to create missing columns with default values
            for col in missing_columns:
                if col == 'Change':
                    df['Change'] = df['Price'].pct_change()
                else:
                    df[col] = df['Price']  # Use Price as default for missing columns
            print("Created missing columns with default values")
            
        # Ensure data is sorted by date
        df = df.sort_values('Date')
        
        # Create target column (next day's price)
        df['target'] = df['Price'].shift(-1)
        
        # 1. Basic Price Features
        # Clean and convert columns to numeric types
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        
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
        df['Volume'] = df['Volume'].apply(parse_volume).fillna(0)

        df['Change'] = pd.to_numeric(df['Change'].astype(str).str.strip().str.replace('%', '').str.replace(',', '.'), errors='coerce').fillna(0)
        
        # 2. Lag Features
        for lag in range(1, n_lags + 1):
            df[f'lag_{lag}'] = df['Price'].shift(lag)
        
        # 3. Price and Return Features
        df['Price_Pct_Change'] = df['Price'].pct_change()
        df['Price_Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # 4. Moving Averages and Trends
        for period in [5, 10, 20]:
            df[f'SMA_{period}'] = df['Price'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Price'].ewm(span=period, adjust=False).mean()
            df[f'Price_SMA_{period}_Dist'] = (df['Price'] - df[f'SMA_{period}']) / df[f'SMA_{period}']
        
        # 5. Volatility Indicators
        for period in [5, 10, 20]:
            df[f'Volatility_{period}'] = df['Price'].rolling(window=period).std() / df['Price'].rolling(window=period).mean()
        
        # 6. Price Range Features
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Price']
        df['Gap'] = (df['Open'] - df['Price'].shift(1)) / df['Price'].shift(1)
        
        # 7. Volume Features
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Pct_Change'] = df['Volume'].pct_change()
        
        # 8. Technical Indicators
        # RSI
        delta = df['Price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Price'].ewm(span=12, adjust=False).mean()
        exp2 = df['Price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 9. Time Features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 5)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 5)
        
        # 10. High-Low Features
        df['HL_Ratio'] = df['High'] / df['Low']
        df['HL_Diff'] = df['High'] - df['Low']
        df['HL_Diff_Pct'] = (df['High'] - df['Low']) / df['Low']
        
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill for lag features
        lag_columns = [col for col in df.columns if col.startswith('lag_')]
        df[lag_columns] = df[lag_columns].fillna(method='ffill')
        
        # Fill remaining NaN with 0
        df = df.fillna(0)
        
        # Remove rows without enough lag information or target
        df = df.dropna(subset=[f'lag_{n_lags}', 'target'])
        
        if len(df) == 0:
            print("Error: No data left after processing")
            return pd.DataFrame()
            
        print(f"Successfully created {len(df)} rows of data with {len(df.columns)} features")
        print(f"Features created: {', '.join(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Error creating features: {str(e)}")
        return pd.DataFrame()

def load_multiple_stocks(file_paths):
    all_data = pd.DataFrame()
    for file_path in file_paths:
        data = load_data(file_path)
        stock_name = os.path.basename(file_path).split('.')[0]
        data = data.rename(columns={'Price': f'Price_{stock_name}', 'Volume': f'Volume_{stock_name}'})
        if all_data.empty:
            all_data = data[['Date', f'Price_{stock_name}', f'Volume_{stock_name}']]
        else:
            all_data = pd.merge(all_data, data[['Date', f'Price_{stock_name}', f'Volume_{stock_name}']], on='Date', how='outer')
    return all_data.sort_values('Date').fillna(method='ffill')

def create_lstm_dataset(data, n_lags=5):
    """
    Prepare data for LSTM model
    """
    X = []
    y = []
    price_values = data['Price'].values
    
    for i in range(n_lags, len(data)):
        X.append(price_values[i-n_lags:i])
        y.append(price_values[i])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape input to [samples, time steps, features] as LSTM requires
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def build_lstm_model(input_shape, units=50):
    """
    Create a simple LSTM model
    """
    if not HAS_TENSORFLOW:
        return None
        
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=units))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

def get_ensemble_model(models_dict):
    """
    Create an ensemble model from existing models
    """
    estimators = [(name, model) for name, model in models_dict.items()]
    ensemble = VotingRegressor(estimators=estimators)
    return ensemble

def predict_with_prophet(df, periods=30):
    """
    Use Facebook Prophet for forecasting
    """
    try:
        from prophet import Prophet
        
        # Prepare data
        prophet_df = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
        
        # Initialize and train model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        
        # Create data for prediction
        future = model.make_future_dataframe(periods=periods)
        
        # Predict
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat']]
    except:
        return None

# Add function to build LightGBM model
def build_lightgbm_model(params=None):
    """
    Create an enhanced LightGBM model with optimized parameters
    """
    default_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 3000,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.01,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'max_depth': 15,
        'random_state': 42
    }
    
    if params:
        default_params.update(params)
    
    return lgb.LGBMRegressor(**default_params)

def find_baodautu_prediction(stock_name, date):
    """
    Tìm giá trị dự báo từ file baodautu_chuyen.csv cho một mã chứng khoán và ngày cụ thể
    Returns: (price, is_from_file) - giá dự báo và cờ đánh dấu dữ liệu lấy từ file
    """
    try:
        # Thông tin debug cho dự báo
        print(f"Tìm dự báo cho {stock_name} vào ngày {date.strftime('%d/%m/%Y')}")
        date_str = date.strftime('%d/%m/%Y')
        
        # Sử dụng đường dẫn từ biến toàn cục
        baodautu_file = BAODAUTU_CHUYEN
        
        # Kiểm tra file tồn tại
        if os.path.exists(baodautu_file):
            try:
                # Đọc dữ liệu file với nhiều cách khác nhau để đảm bảo đọc được
                df = None
                
                # Thử phương pháp 1: Đọc với on_bad_lines='skip'
                try:
                    df = pd.read_csv(baodautu_file, encoding='utf-8', on_bad_lines='skip')
                except (TypeError, pd.errors.ParserError):
                    # Thử phương pháp 2: Đọc với error_bad_lines=False (pandas cũ)
                    try:
                        df = pd.read_csv(baodautu_file, encoding='utf-8', error_bad_lines=False)
                    except (TypeError, pd.errors.ParserError):
                        # Thử phương pháp 3: Đọc với delimiters khác nhau
                        for delimiter in [',', ';', '\t']:
                            try:
                                df = pd.read_csv(baodautu_file, encoding='utf-8', sep=delimiter, on_bad_lines='skip')
                                if len(df.columns) > 1:  # Nếu có ít nhất 2 cột
                                    break
                            except:
                                continue
                                
                if df is None or df.empty:
                    print(f"Không thể đọc file {baodautu_file} với bất kỳ phương pháp nào")
                    return None, False
                
                # Debug: hiển thị các cột của file CSV
                print(f"Các cột trong file: {df.columns.tolist()}")
                print(f"Số dòng trong file: {len(df)}")
                
                # Chuẩn bị dữ liệu
                # 1. Tạo cột ngày (hoặc sử dụng cột đầu tiên nếu chứa ngày)
                try:
                    # Thử chuyển đổi cột đầu thành ngày tháng
                    df['Date'] = pd.to_datetime(df.iloc[:, 0], format='%d/%m/%Y %H:%M', errors='coerce')
                    
                    # Nếu không đọc được, thử thêm một số định dạng khác
                    if df['Date'].isna().all():
                        for date_format in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                            try:
                                df['Date'] = pd.to_datetime(df.iloc[:, 0], format=date_format, errors='coerce')
                                if not df['Date'].isna().all():
                                    break
                            except:
                                continue
                                
                    # Nếu vẫn không đọc được ngày tháng
                    if df['Date'].isna().all():
                        print(f"Không thể đọc định dạng ngày tháng trong file {baodautu_file}")
                        return None, False
                except Exception as e:
                    print(f"Lỗi xử lý dữ liệu ngày tháng: {e}")
                    return None, False
                
                # 2. Lọc dữ liệu theo ngày tháng và mã chứng khoán
                try:
                    # Lọc theo ngày
                    df_filtered = df[df['Date'].dt.strftime('%d/%m/%Y') == date_str]
                    
                    if df_filtered.empty:
                        print(f"Không tìm thấy dữ liệu cho ngày {date_str}")
                        return None, False
                    
                    print(f"Tìm thấy {len(df_filtered)} dòng dữ liệu cho ngày {date_str}")
                    
                    # Xử lý trường hợp đặc biệt cho HNX-Index và VN Index
                    if stock_name == "HNX-Index" or stock_name == "VN Index":
                        # Tìm kiếm tất cả các biến thể của tên chỉ số
                        stock_variants = []
                        if stock_name == "HNX-Index":
                            stock_variants = ["HNX-Index", "HNX", "HNXIndex", "HNX Index", "HNX-INDEX"]
                        elif stock_name == "VN Index":
                            stock_variants = ["VN Index", "VN-Index", "VNIndex", "VNINDEX", "VN"]
                        
                        # In thông tin debug
                        print(f"Tìm kiếm chỉ số {stock_name} với các biến thể: {stock_variants}")
                        for idx, row in df_filtered.iterrows():
                            print(f"Dòng {idx}: {row.tolist()}")
                        
                        # Tìm mã chứng khoán trong tất cả các cột
                        stock_matches = []
                        for variant in stock_variants:
                            for col_idx in range(min(5, len(df_filtered.columns))):  # Kiểm tra tất cả các cột
                                if col_idx < len(df_filtered.columns):
                                    # Chuyển đổi cả hai thành chuỗi để so sánh linh hoạt hơn
                                    column_values = df_filtered.iloc[:, col_idx].astype(str)
                                    matches = df_filtered[column_values.str.contains(variant, case=False, na=False)]
                                    if not matches.empty:
                                        print(f"Tìm thấy {len(matches)} kết quả khớp với '{variant}' trong cột {col_idx}")
                                        stock_matches.append(matches)
                    else:
                        # Lọc theo mã chứng khoán như cũ
                        stock_matches = []
                        
                        # Kiểm tra xem cột nào chứa mã chứng khoán
                        for col_idx in range(1, min(5, len(df.columns))): # Kiểm tra 4 cột đầu tiên sau cột ngày
                            if col_idx < len(df.columns):
                                # So sánh cả chính xác và contains
                                matches = df_filtered[df_filtered.iloc[:, col_idx] == stock_name]
                                if not matches.empty:
                                    stock_matches.append(matches)
                    
                    # Ghép lại các kết quả tìm được
                    if stock_matches:
                        df_final = pd.concat(stock_matches).drop_duplicates()
                        print(f"Tìm thấy tổng cộng {len(df_final)} dòng khớp với mã {stock_name}")
                    else:
                        print(f"Không tìm thấy mã {stock_name} trong file cho ngày {date_str}")
                        return None, False
                    
                    # 3. Lấy giá dự báo từ cột cuối cùng (hoặc cột chứa giá)
                    if not df_final.empty:
                        # Hiển thị tất cả dữ liệu tìm được để debug
                        for idx, row in df_final.iterrows():
                            print(f"Dòng {idx} tìm được: {row.tolist()}")
                            
                        # Lấy giá trị từ cột cuối
                        price_val = df_final.iloc[0, -1]
                        
                        # Xử lý các kiểu dữ liệu khác nhau
                        try:
                            if isinstance(price_val, str):
                                # Nếu là chuỗi, thay dấu phẩy bằng dấu chấm
                                price_val = price_val.replace(',', '.')
                                price = float(price_val)
                            elif isinstance(price_val, (int, float)):
                                # Nếu đã là số, giữ nguyên
                                price = float(price_val)
                            elif isinstance(price_val, pd.Timestamp):
                                # Nếu là timestamp, có thể cột bị chọn sai, thử cột khác
                                for col_idx in range(2, len(df_final.columns)):
                                    val = df_final.iloc[0, col_idx]
                                    if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '', 1).replace(',', '', 1).isdigit()):
                                        try:
                                            if isinstance(val, str):
                                                val = val.replace(',', '.')
                                            price = float(val)
                                            break
                                        except:
                                            continue
                                else:
                                    print(f"Không tìm thấy giá trị hợp lệ cho dự báo")
                                    return None, False
                            else:
                                # Trường hợp khác, cố gắng chuyển đổi
                                price = float(price_val)
                            
                            print(f"Đã tìm thấy dự báo từ file cho {stock_name} vào ngày {date_str}: {price}")
                            return price, True
                        except (ValueError, TypeError) as e:
                            print(f"Lỗi chuyển đổi giá trị '{price_val}' sang số: {e}")
                except Exception as e:
                    print(f"Lỗi lọc dữ liệu: {e}")
            except Exception as e:
                print(f"Lỗi xử lý file baodautu_chuyen.csv: {e}")
        else:
            print(f"File {baodautu_file} không tồn tại")
            
            # Kiểm tra các vị trí khác
            possible_locations = [
                './baodautu_chuyen.csv',
                '../baodautu_chuyen.csv',
                'D:/CODE/GIT_AI/Workplace/Stock/Baodautu/baodautu_chuyen.csv',
                os.path.join(os.getcwd(), 'Baodautu/baodautu_chuyen.csv'),
                os.path.join(os.getcwd(), 'baodautu_chuyen.csv')
            ]
            
            for loc in possible_locations:
                print(f"Kiểm tra: {loc}")
                if os.path.exists(loc):
                    print(f"Đã tìm thấy file tại: {loc}")
                    # Đọc file từ vị trí mới một lần
                    df = pd.read_csv(loc, encoding='utf-8', on_bad_lines='skip')
                    if not df.empty:
                        print(f"Đã đọc thành công file từ: {loc}")
                        baodautu_file = loc
                        # Gọi lại hàm này một lần nữa
                        result = find_baodautu_prediction(stock_name, date)
                        return result
        
        # Trả về None nếu không tìm thấy dữ liệu
        return None, False
    except Exception as e:
        print(f"Lỗi đọc dữ liệu dự báo: {e}")
        return None, False

def get_train_files():
    """
    Get list of training files from the train directory
    Returns:
        list: List of full paths to training files
    """
    print(f"\nĐang tìm file train trong thư mục: {os.path.abspath(TRAIN_DIR)}")
    
    if not os.path.exists(TRAIN_DIR):
        print("Thư mục Train không tồn tại!")
        return []
    
    train_files = []
    for file in os.listdir(TRAIN_DIR):
        if file.endswith('.csv'):
            full_path = os.path.join(TRAIN_DIR, file)
            train_files.append(full_path)
            print(f"Đã tìm thấy file train: {file}")
    
    if not train_files:
        print("Không tìm thấy file .csv nào trong thư mục Train!")
    else:
        print(f"Tìm thấy {len(train_files)} file train")
    
    return train_files

def clear_models_dir():
    """
    Clear all files in the models directory
    """
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            file_path = os.path.join(MODELS_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(MODELS_DIR)
        print(f"Created new models directory at: {MODELS_DIR}")

def save_feature_names(feature_names, stock_name):
    """
    Save feature names used during training
    """
    try:
        filename = f"{stock_name}_features.joblib"
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(feature_names, filepath)
        print(f"Saved feature names for {stock_name}")
        return True
    except Exception as e:
        print(f"Error saving feature names for {stock_name}: {e}")
        return False

def load_feature_names(stock_name):
    """
    Load feature names used during training
    """
    try:
        filename = f"{stock_name}_features.joblib"
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Feature names file not found for {stock_name}")
            return None
        feature_names = joblib.load(filepath)
        print(f"Loaded feature names for {stock_name}")
        return feature_names
    except Exception as e:
        print(f"Error loading feature names for {stock_name}: {e}")
        return None

def retrain_all_models():
    """
    Retrain all models for each stock
    """
    print("Starting model retraining process...")

    # Clear models directory
    print("\nClearing models directory...")
    clear_models_dir()

    # Get list of training files
    train_files = get_train_files()
    
    if not train_files:
        print("No training files found!")
        return

    print(f"\nStarting training for {len(train_files)} stocks...\n")

    for file_path in train_files:
        try:
            stock_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"\nProcessing {stock_name}...")
            
            # Load and preprocess data
            data = load_data(file_path)
            if data is None or data.empty:
                print(f"Error: Could not load data for {stock_name}")
                continue

            # Create features
            features = create_advanced_features(data)
            if features is None or features.empty:
                print(f"Error creating features for {stock_name}")
                continue

            # Get feature columns for training (exclude Date and target)
            feature_columns = [col for col in features.columns if col not in ['Date', 'target']]
            
            # Save feature order
            save_feature_order(feature_columns, stock_name)
            
            # Split features and target
            X = features[feature_columns]
            y = features['target']

            # Split into train and validation sets
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Save scaler
            save_scaler(scaler, stock_name)

            # Convert to DataFrame to preserve feature names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns)

            # 1. RandomForest with optimized parameters
            print(f"Training RandomForest model for {stock_name}...")
            rf_model = RandomForestRegressor(
                n_estimators=3000,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                n_jobs=-1,
                random_state=42
            )
            rf_model.fit(X_train_scaled, y_train)
            save_model(rf_model, 'rf', stock_name)
            
            # 2. XGBoost with optimized parameters
            print(f"Training XGBoost model for {stock_name}...")
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                colsample_bytree=0.8,
                learning_rate=0.02,
                max_depth=8,
                alpha=2,
                n_estimators=3000,
                subsample=0.8,
                min_child_weight=3
            )
            xgb_model.fit(X_train_scaled, y_train)
            save_model(xgb_model, 'xgb', stock_name)
            
            # 3. LightGBM with optimized parameters
            print(f"Training LightGBM model for {stock_name}...")
            lgb_model = build_lightgbm_model()
            lgb_model.fit(X_train_scaled, y_train)
            save_model(lgb_model, 'lgb', stock_name)

            # 4. Decision Tree
            print(f"Training Decision Tree model for {stock_name}...")
            dt_model = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=4,
                random_state=42
            )
            dt_model.fit(X_train_scaled, y_train)
            save_model(dt_model, 'dt', stock_name)

            print(f"Successfully trained all models for {stock_name}")

        except Exception as e:
            print(f"Error processing {stock_name}: {str(e)}")
            continue

    print("\nCompleted retraining all models!")

if __name__ == "__main__":
    print("Starting model retraining process...")
    retrain_all_models()
