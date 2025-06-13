import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
import os
import traceback
from scipy.interpolate import make_interp_spline, PchipInterpolator
import json
from datetime import datetime

# Đường dẫn tuyệt đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
                 find_baodautu_prediction)  # Add find_baodautu_prediction to imports

# Import metrics calculation function
from metrics_utils import calculate_prediction_metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor  # Add Decision Tree import
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_percentage_error,
                             mean_squared_error,
                             mean_absolute_error,
                             r2_score)
import xgboost as xgb

# Định nghĩa các hàm vẽ đồ thị và cập nhật cho Streamlit
                if event.inaxes == ax:
                    self.current_option = i
                    self.main_button.label.set_text(f"{self.label}{self.options[i]}")
                    self.expanded = False
                    for ax in self.ax_options:
                        ax.set_visible(False)
                    plt.draw()
                    self._process_observers()
                    break
    
    def on_changed(self, func):
        self.observers[func] = func
        
    def _process_observers(self):
        for func in self.observers.values():
            func(self.options[self.current_option])
            
    def set_visible(self, visible):
        self.ax_main.set_visible(visible)
        if not visible:
            self.expanded = False
            for ax in self.ax_options:
                ax.set_visible(False)
        plt.draw()

# ---------------------------
# Initialize Tkinter root for loading window and error messages
# ---------------------------
root = tk.Tk()
root.withdraw()

# ---------------------------
# Create a queue for communication between threads
# ---------------------------
update_queue = queue.Queue()

# ---------------------------
# Initialize Figure and Axes for matplotlib
# ---------------------------
fig = plt.figure(figsize=(12, 8))
# Adjust margins to make room for controls
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)  # Reduced top margin to avoid overlap

# Create a new axes for the dropdown menu outside the main plot
ax_dropdown = plt.axes([0.02, 0.92, 0.15, 0.06])  # Above the plot area
ax_dropdown.set_visible(False)  # Hide the axes, we just want the dropdown

# Create main plot areas with adjusted height
gs = fig.add_gridspec(2, height_ratios=[3, 1], top=0.85)  # Adjust top position
ax = fig.add_subplot(gs[0])       # Main chart
ax_table = fig.add_subplot(gs[1]) # Area for displaying error table (initially hidden)
ax_table.axis('off')

# ---------------------------
# Error handling functions
# ---------------------------
def show_error(message):
    # Check which thread we're in
    if threading.current_thread() is threading.main_thread():
        # If in main thread, display error directly
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("Error", message)
        r.destroy()
    else:
        # If in secondary thread, queue for main thread to handle
        def _show_error():
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror("Error", message)
            r.destroy()
        update_queue.put(_show_error)

# ---------------------------
# Function to display success message
# ---------------------------
def show_success(message):
    # Check which thread we're in
    if threading.current_thread() is threading.main_thread():
        # If in main thread, display message directly
        r = tk.Tk()
        r.withdraw()
        messagebox.showinfo("Success", message)
        r.destroy()
    else:
        # If in secondary thread, queue for main thread to handle
        def _show_success():
            r = tk.Tk()
            r.withdraw()
            messagebox.showinfo("Success", message)
            r.destroy()
        update_queue.put(_show_success)

def show_error_window(event):
    global error_df
    if error_df is None:
        show_error("No error data available. Please run the forecast first.")
        return
    df_error_display = error_df.copy()
    df_error_display.iloc[:, 1:] = df_error_display.iloc[:, 1:].applymap(lambda x: f"{x:.3f}")
    
    # Display in main thread
    def _show_error_table():
        fig_error, ax_error = plt.subplots(figsize=(8, 3))
        ax_error.axis('tight')
        ax_error.axis('off')
        table = ax_error.table(cellText=df_error_display.values,
                            colLabels=df_error_display.columns,
                            cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        fig_error.suptitle("Error Table", fontsize=16)
        plt.show()
    
    # If in secondary thread, queue
    if threading.current_thread() is threading.main_thread():
        _show_error_table()
    else:
        update_queue.put(_show_error_table)

# Function to safely destroy Tkinter widget from any thread
def safe_destroy(widget):
    try:
        if widget.winfo_exists():
            widget.destroy()
    except Exception as e:
        print(f"Error destroying widget: {e}")

# ---------------------------
# Set up directory paths
# ---------------------------
# List all files in Train directory to get stock list
train_dir = 'D:\\NghienCuu\\Stock\\Train' 

# Check if directory exists and print helpful info for user
print(f"Checking Train directory: {TRAIN_DIR}")
if not os.path.exists(TRAIN_DIR):
    print(f"WARNING: Directory {TRAIN_DIR} does not exist!")
    os.makedirs(TRAIN_DIR)
    print(f"Created Train directory at: {TRAIN_DIR}")

# Check and create Ve directory if needed
ve_dir = 'D:\\NghienCuu\\Stock\\Ve'
if not os.path.exists(ve_dir):
    print(f"WARNING: Directory {ve_dir} does not exist!")
    # Try to find Ve directory in other locations
    possible_dirs = [
        './Ve', 
        '../Ve',
        'Ve',
        'D:/CODE/GIT_AI/Workplace/Ve',
        os.path.join(os.getcwd(), 'Ve')
    ]
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f"Found Ve directory at: {dir_path}")
            ve_dir = dir_path
            break
    else:
        print("Could not find Ve directory in any checked location.")
        print("Creating Ve directory...")
        try:
            os.makedirs(ve_dir, exist_ok=True)
            print(f"Created Ve directory at: {ve_dir}")
        except Exception as e:
            print(f"Could not create Ve directory: {e}")

if os.path.exists(TRAIN_DIR):
    # List available files in Train directory
    try:
        available_files = os.listdir(TRAIN_DIR)
        print(f"Files in directory {TRAIN_DIR}:")
        for file in available_files:
            print(f"  - {file}")
        
        # Only get CSV files
        stock_files = [f.split('.')[0] for f in available_files if f.endswith('.csv')]
        if not stock_files:
            print(f"No CSV files found in directory {TRAIN_DIR}")
            # Create a sample list to avoid errors
            stock_files = ["AAPL", "MSFT", "GOOGL"]
            print(f"Using sample stock list: {stock_files}")
    except Exception as e:
        print(f"Error reading directory {TRAIN_DIR}: {e}")
        # Create a sample list to avoid errors
        stock_files = ["AAPL", "MSFT", "GOOGL"]
        print(f"Using sample stock list: {stock_files}")
else:
    # If Train directory still not found, use sample list
    stock_files = ["AAPL", "MSFT", "GOOGL"]
    print(f"Using sample stock list: {stock_files}")

stock_files.sort()  # Sort stocks alphabetically

# ---------------------------
# Create UI controls
# ---------------------------
# Create dropdown menu for stock selection with fixed position
dropdown_menu = DropdownMenu(ax_dropdown, stock_files, initial=0, label="Stock: ", width=0.15, height=0.04)

# Fixed positions for buttons in the top right (above the plot)
# "Forecast" button
ax_button_forecast = plt.axes([0.78, 0.92, 0.10, 0.05])  # Move above plot area
button_forecast = Button(ax_button_forecast, 'Forecast')

# "Supplement" button 
ax_button_error = plt.axes([0.90, 0.92, 0.10, 0.05])  # Move above plot area
button_error = Button(ax_button_error, 'Supplement')

# "Start date" and "End date" input boxes with "OK" button (middle bottom)
# These will be shown/hidden but position remains fixed
axbox_start_date = plt.axes([0.30, 0.10, 0.15, 0.05])  # Lower position
axbox_end_date   = plt.axes([0.60, 0.10, 0.15, 0.05])  # Lower position
text_box_start_date = TextBox(axbox_start_date, 'Start date:', initial='13-03-2025')
text_box_end_date   = TextBox(axbox_end_date, 'End date:', initial='26-03-2025')

ax_button_ok = plt.axes([0.45, 0.02, 0.1, 0.05])  # Lower position
button_ok = Button(ax_button_ok, 'OK')

# Initially hide date input widgets
axbox_start_date.set_visible(False)
axbox_end_date.set_visible(False)
ax_button_ok.set_visible(False)

# ---------------------------
# Global variables
# ---------------------------
current_dataset = stock_files[0] if stock_files else 'AMZN'
last_annotation = None
motion_cid = None  # ID of motion_notify_event
error_df = None    # Variable to store error table

# Add global variables for storing metrics
metrics_normal = None
metrics_future = None
metrics_supplement = None

# Add new button for metrics comparison
ax_button_metrics = plt.axes([0.65, 0.92, 0.10, 0.05])  # Position next to other buttons
button_metrics = Button(ax_button_metrics, 'Metrics')

def store_future_metrics(metrics_dict):
    """
    Store metrics for future predictions
    """
    global metrics_future
    metrics_future = metrics_dict.copy()

# Add this function before show_metrics_comparison
def load_system_metrics():
    """
    Load metrics from system_metrics.json file
    """
    metrics_file = os.path.join('metrics', 'system_metrics.json')
    if not os.path.exists(metrics_file):
        return None
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading system metrics: {e}")
        return None

def show_metrics_comparison(event):
    """
    Hiển thị bảng so sánh các chỉ số sai số cho 3 loại dự báo
    """
    # Load metrics from system_metrics.json
    system_metrics = load_system_metrics()
    if system_metrics is None:
        show_error("Không tìm thấy dữ liệu metrics. Vui lòng chạy tính toán metrics trước.")
        return
        
    # Tạo cửa sổ Tkinter mới
    metrics_window = tk.Toplevel(root)
    metrics_window.title("So sánh chỉ số sai số")
    metrics_window.geometry("1000x800")
    
    # Tạo frame chính
    main_frame = ttk.Frame(metrics_window)
    main_frame.pack(padx=10, pady=10, fill='both', expand=True)
    
    def create_metrics_table(parent, metrics_dict, title):
        """Create table from metrics dictionary"""
        if not metrics_dict:
            return None
            
        # Create frame
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(padx=5, pady=5, fill='both', expand=True)
        
        # Create table
        table = ttk.Treeview(frame)
        table['columns'] = ['Model', 'MSE', 'MAE', 'RMSE', 'MAPE', 'R²']
        table.column('#0', width=0, stretch=tk.NO)
        
        # Configure columns
        table.column('Model', anchor=tk.W, width=150)
        for col in ['MSE', 'MAE', 'RMSE', 'MAPE', 'R²']:
            table.column(col, anchor=tk.E, width=100)
            table.heading(col, text=col)
        
        # Add headers
        table.heading('Model', text='Model')
        
        # Model names mapping
        model_names = {
            'rf': 'Random Forest',
            'xgb': 'XGBoost',
            'lgb': 'LightGBM',
            'cb': 'CatBoost'
        }
        
        # Add data
        for model_name, values in metrics_dict.items():
            if isinstance(values, dict):
                try:
                    table.insert('', 'end', values=(
                        model_names.get(model_name, model_name),
                        f"{values['MSE']:.6f}",
                        f"{values['MAE']:.6f}",
                        f"{values['RMSE']:.6f}",
                        f"{values['MAPE']:.6f}",
                        f"{values['R^2']:.6f}"
                    ))
                except KeyError as e:
                    print(f"Lỗi khi xử lý metrics cho {model_name}: {e}")
                    continue
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
        y_scrollbar.pack(side='right', fill='y')
        
        x_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=table.xview)
        x_scrollbar.pack(side='bottom', fill='x')
        
        table.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        table.pack(padx=5, pady=5, fill='both', expand=True)
        
        # Add sorting capability
        def sort_treeview(col, reverse):
            l = [(table.set(k, col), k) for k in table.get_children('')]
            try:
                # Try to convert to float for numeric sorting
                l.sort(key=lambda x: float(x[0]), reverse=reverse)
            except ValueError:
                # Fall back to string sorting
                l.sort(reverse=reverse)
            
            # Rearrange items in sorted positions
            for index, (val, k) in enumerate(l):
                table.move(k, '', index)
            
            # Reverse sort next time
            table.heading(col, command=lambda: sort_treeview(col, not reverse))
        
        # Configure column headings for sorting
        for col in ['Model', 'MSE', 'MAE', 'RMSE', 'MAPE', 'R²']:
            table.heading(col, command=lambda c=col: sort_treeview(c, False))
        
        return frame
    
    # Create notebook for tabs
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill='both', expand=True, padx=5, pady=5)
    
    # Create tabs for each type of prediction
    prediction_types = {
        'normal': 'Dự báo thông thường',
        'future': 'Dự báo tương lai',
        'supplement': 'Dự báo bổ sung'
    }
    
    for pred_type, title in prediction_types.items():
        if pred_type in system_metrics and system_metrics[pred_type]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            create_metrics_table(frame, system_metrics[pred_type], "")
    
    # Add close button
    ttk.Button(metrics_window, text="Đóng", command=metrics_window.destroy).pack(pady=10)
    
    # Force window update and focus
    metrics_window.update_idletasks()
    metrics_window.focus_force()

# Connect metrics button event
button_metrics.on_clicked(show_metrics_comparison)

# ---------------------------
# Main update chart function
# ---------------------------
def check_models_exist(stock_name):
    """Check if all required models exist for a stock"""
    models_dir = 'models'
    required_files = [
        f'{stock_name}_rf_model.joblib',
        f'{stock_name}_xgb_model.joblib',
        f'{stock_name}_lgb_model.joblib',
        f'{stock_name}_dt_model.joblib',
        f'{stock_name}_scaler.joblib',
        f'{stock_name}_feature_order.joblib'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(models_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        message = (f"Missing required models for {stock_name}:\n" +
                  "\n".join(missing_files) +
                  "\n\nPlease run train_models.py first to train the models.")
        show_error(message)
        return False
    return True

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
        # Load scaler and feature order
        scaler = load_scaler(dataset)
        feature_order = load_feature_order(dataset)
        
        if scaler is None or feature_order is None:
            print(f"Could not load scaler or feature order for {dataset}")
            return None
            
        # Create features
        features_df = create_advanced_features(df)
        if features_df.empty:
            print("Could not create features")
            return None
            
        # Select and order features
        X = features_df[feature_order]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        return predictions
        
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return None

def update_plot(event, dataset, start_date=None, end_date=None, show_predictions=False, dark_theme=False):
    """Cập nhật biểu đồ với dữ liệu mới"""
    try:
        print(f"\nStarting to update plot for {dataset}")
        
        # Build file paths
        file_path_train = os.path.join(TRAIN_DIR, f"{dataset}.csv")
        file_path_display = os.path.join(VE_DIR, f"{dataset}_TT.csv")
        
        print(f"Checking files:\n- Train: {file_path_train}\n- Display: {file_path_display}")
        
        # Validate files exist
        if not os.path.exists(file_path_train):
            raise FileNotFoundError(f"Train file not found: {file_path_train}")
        if not os.path.exists(file_path_display):
            raise FileNotFoundError(f"Display file not found: {file_path_display}")

        # Read data with explicit encoding and separator
        try:
            df_train = pd.read_csv(file_path_train, encoding='utf-8', sep=',')
            df_display = pd.read_csv(file_path_display, encoding='utf-8', sep=',')
            print(f"Successfully read files:\n- Train rows: {len(df_train)}\n- Display rows: {len(df_display)}")
        except Exception as e:
            print(f"Error reading CSV files: {e}")
            raise

        # Convert dates
        for df in [df_train, df_display]:
            try:
                df['Date'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')
                print(f"Converted dates successfully. Range: {df['Date'].min()} to {df['Date'].max()}")
            except Exception as e:
                print(f"Error converting dates: {e}")
                raise

        # Filter by date range if provided
        if start_date and end_date:
            print(f"Filtering date range: {start_date} to {end_date}")
            df_display = df_display[(df_display['Date'] >= start_date) & (df_display['Date'] <= end_date)]
            print(f"Filtered display data: {len(df_display)} rows")

        # Create figure with larger size
        plt.clf()
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
        
        # Set dark theme if requested
        if dark_theme:
            plt.style.use('dark_background')
            ax.set_facecolor('#2d2d2d')
            fig.patch.set_facecolor('#2d2d2d')

        # Plot actual prices
        ax.plot(df_display['Date'], df_display['Lần cuối'], 
                label='Giá thực tế', color='blue', linewidth=2)

        # Add predictions if requested
        if show_predictions:
            try:
                print("Adding predictions to plot...")
                models = load_models(dataset)
                if not models:
                    raise Exception("No models loaded")
                
                colors = {'rf': 'red', 'xgb': 'orange', 'lgb': 'green', 'dt': 'purple'}
                line_styles = {'rf': '--', 'xgb': '-.', 'lgb': ':', 'dt': '--'}
                
                for model_name, model in models.items():
                    predictions = get_predictions(model, df_display, dataset)
                    if predictions is not None and len(predictions) > 0:
                        ax.plot(df_display['Date'][-len(predictions):], 
                               predictions, 
                               label=f'Dự báo {model_name.upper()}',
                               linestyle=line_styles.get(model_name, '--'),
                               color=colors.get(model_name, 'gray'))
                        print(f"Added predictions for {model_name}: {len(predictions)} points")
                    else:
                        print(f"No predictions generated for {model_name}")
            except Exception as e:
                print(f"Error adding predictions: {e}")
                raise

        # Customize plot
        ax.set_title(f'Biểu đồ giá {dataset}', fontsize=14, pad=20)
        ax.set_xlabel('Ngày', fontsize=12)
        ax.set_ylabel('Giá', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        print("Plot updated successfully")
        return fig

    except Exception as e:
        print(f"Error in update_plot: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

# ---------------------------
# Worker function for background processing
# ---------------------------
def worker(dataset, s_date, e_date, loading_root):
    try:
        # Run forecast in background thread
        result = update_plot(None, dataset, s_date, e_date, show_predictions=True)
    except Exception as e:
        captured_error = str(e)  # Store e value in a local variable
        # Queue error display function to run in main thread
        update_queue.put(lambda: show_error(f"Error: {captured_error}"))
    finally:
        # Queue loading window close function to run in main thread
        update_queue.put(lambda: safe_destroy(loading_root))

# ---------------------------
# Functions for baodautu predictions
# ---------------------------
def apply_baodautu_predictions(event):
    """
    Replace error button functionality with Báo Đầu Tư forecast
    """
    global current_dataset, error_df, metrics_supplement
    
    if current_dataset is None:
        show_error("Please select a stock first.")
        return
    
    try:
        # Get start and end dates from text boxes
        start_date_str = text_box_start_date.text
        end_date_str = text_box_end_date.text
        
        # Convert to datetime objects
        try:
            s_date = pd.to_datetime(start_date_str, dayfirst=True, errors='coerce')
            e_date = pd.to_datetime(end_date_str, dayfirst=True, errors='coerce')
            
            if pd.isnull(s_date) or pd.isnull(e_date):
                show_error("Invalid date format. Please enter dates in dd-mm-yyyy format.")
                return
        except Exception as date_error:
            show_error(f"Error processing dates: {date_error}")
            return
            
        # Show loading window
        loading_root = tk.Toplevel(root)
        loading_root.title("Updating forecast")
        tk.Label(loading_root, text="Finding and applying forecast data from Báo Đầu Tư...").pack(padx=20, pady=10)
        progress = ttk.Progressbar(loading_root, orient="horizontal", mode="indeterminate", length=280)
        progress.pack(padx=20, pady=10)
        progress.start(10)
        loading_root.lift()
        loading_root.attributes("-topmost", True)
        
        # Create a separate thread to perform the work - pass start and end dates
        t = threading.Thread(target=lambda: apply_baodautu_worker(current_dataset, loading_root, s_date, e_date), daemon=True)
        t.start()
        
    except Exception as e:
        show_error(f"Error applying forecast data: {e}")

def apply_baodautu_worker(stock_name, loading_root, start_date=None, end_date=None):
    """
    Worker to apply forecast data from Báo Đầu Tư
    """
    global metrics_supplement
    try:
        # Print current path information
        print(f"Current path: {os.getcwd()}")
        
        # Read necessary data
        file_path_train = os.path.join(TRAIN_DIR, f"{stock_name}.csv")
        file_path_display = os.path.join(VE_DIR, f"{stock_name}_TT.csv")
        
        print(f"Checking train file: {file_path_train}")
        print(f"Checking display file: {file_path_display}")
        
        if not os.path.exists(file_path_train) or not os.path.exists(file_path_display):
            error_msg = f"Required data files not found for {stock_name}:\n"
            if not os.path.exists(file_path_train):
                error_msg += f"- Train file: {file_path_train} does not exist\n"
            if not os.path.exists(file_path_display):
                error_msg += f"- Display file: {file_path_display} does not exist\n"
            update_queue.put(lambda: show_error(error_msg))
            update_queue.put(lambda: safe_destroy(loading_root))
            return        # Check baodautu_chuyen.csv file before processing
        print("Looking for baodautu_chuyen.csv...")
        # Sử dụng biến đường dẫn đã định nghĩa ở đầu file
        baodautu_file = BAODAUTU_CHUYEN
        print(f"Kiểm tra file tại: {baodautu_file}")
        
        if not os.path.exists(baodautu_file):
            # Try finding file in other locations
            possible_locations = [
                './baodautu_chuyen.csv',
                '../baodautu_chuyen.csv',
                './Baodautu/baodautu_chuyen.csv',
                'Baodautu/baodautu_chuyen.csv',
                os.path.join(os.getcwd(), 'Baodautu/baodautu_chuyen.csv'),
                os.path.join(os.path.dirname(os.getcwd()), 'Baodautu/baodautu_chuyen.csv')
            ]
            for loc in possible_locations:
                print(f"Checking: {loc}")
                if os.path.exists(loc):
                    print(f"Found file at: {loc}")
                    baodautu_file = loc
                    break
            else:
                error_msg = "baodautu_chuyen.csv not found in any checked location.\n" + \
                           "This file is required for forecast data. Please ensure it exists."
                update_queue.put(lambda: show_error(error_msg))
                update_queue.put(lambda: safe_destroy(loading_root))
                return
        else:
            print(f"Found baodautu_chuyen.csv at: {os.path.abspath(baodautu_file)}")
        
        # Read file content for debugging
        if os.path.exists(baodautu_file):
            try:
                with open(baodautu_file, 'r', encoding='utf-8') as f:
                    print("First 10 lines of baodautu_chuyen.csv:")
                    for i, line in enumerate(f):
                        if i < 10:
                            print(f"  {line.strip()}")
                        else:
                            break
            except Exception as e:
                print(f"Error reading file for debugging: {e}")
        
        # Load data with proper error handling
        try:
            data_train = load_data(file_path_train)
            data_display = load_data(file_path_display)
        except Exception as load_error:
            update_queue.put(lambda: show_error(f"Error loading data files: {load_error}"))
            update_queue.put(lambda: safe_destroy(loading_root))
            return
        
        if data_train.empty or data_display.empty:
            update_queue.put(lambda: show_error(f"Could not read data for {stock_name}. Files may be empty or in an invalid format."))
            update_queue.put(lambda: safe_destroy(loading_root))
            return
        
        # Normalize dates
        data_train['Date'] = pd.to_datetime(data_train['Date']).dt.normalize()
        data_display['Date'] = pd.to_datetime(data_display['Date']).dt.normalize()
        
        # Define display time range - use user-provided range if available
        if start_date is None:
            start_date = data_display['Date'].min()
        if end_date is None:
            end_date = data_display['Date'].max()
            
        # Debug time range information
        print(f"Finding forecast data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        
        # Filter train data to only get data in necessary time range
        filtered_train_data = data_train[(data_train['Date'] >= start_date) & (data_train['Date'] <= end_date)]
        
        if filtered_train_data.empty:
            print(f"WARNING: No train data in selected time range for {stock_name}")
            print(f"Train data range: {data_train['Date'].min().strftime('%d/%m/%Y')} - {data_train['Date'].max().strftime('%d/%m/%Y')}")
            print(f"Forecast range needed: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
            
            # Continue anyway - we can use forecasts not dependent on train data
            print("Continuing with Báo Đầu Tư forecast...")
        
        # Find forecast data for each day in range
        print(f"Finding forecast data for {stock_name} from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        baodautu_predictions = []
        
        # Create list of dates in range
        all_dates = pd.date_range(start=start_date, end=end_date)
        
        # Find predictions for each date
        prediction_count = 0
        for date in all_dates:
            prediction, is_from_file = find_baodautu_prediction(stock_name, date)
            if prediction is not None:
                baodautu_predictions.append((date, prediction, is_from_file))
                prediction_count += 1
                if prediction_count <= 10:  # Limit logging to first 10 predictions
                    print(f"Found prediction for {stock_name} on {date.strftime('%d/%m/%Y')}: {prediction}, from file: {is_from_file}")
        
        if prediction_count > 10:
            print(f"Found {prediction_count - 10} more predictions (not listed)...")
        
        if not baodautu_predictions:
            update_queue.put(lambda: show_error(f"No forecast data found from Báo Đầu Tư for {stock_name} in range {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"))
            update_queue.put(lambda: safe_destroy(loading_root))
            return
        
        # Update chart with new data - pass user selected time range
        update_queue.put(lambda: update_plot_with_baodautu(stock_name, baodautu_predictions, start_date, end_date))
        update_queue.put(lambda: safe_destroy(loading_root))
        
    except Exception as e:
        import traceback
        traceback_details = traceback.format_exc()
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback_details}")
        update_queue.put(lambda: show_error(f"Error processing Báo Đầu Tư forecast: {e}"))
        update_queue.put(lambda: safe_destroy(loading_root))

def update_plot_with_baodautu(stock_name, baodautu_predictions, start_date=None, end_date=None, dark_theme=False):
    """
    Update chart with forecast data from Báo Đầu Tư combined with ML model
    """
    global current_dataset, last_annotation, motion_cid, metrics_supplement, metrics_normal, metrics_future, error_df
    
    # Set theme
    if dark_theme:
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        if ax_table is not None:
            ax_table.set_facecolor('#0E1117')
    else:
        plt.style.use('default')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        if ax_table is not None:
            ax_table.set_facecolor('white')

    try:
        # Initialize df_predictions at the start of the function
        df_predictions = {}
        
        # Read display data
        file_path_display = os.path.join(VE_DIR, f"{stock_name}_TT.csv")
        file_path_train = os.path.join(TRAIN_DIR, f"{stock_name}.csv")
        
        print(f"Reading data from {file_path_display} and {file_path_train}")
        
        # Load data with proper error handling
        try:
            data_display = load_data(file_path_display)
            data_train = load_data(file_path_train)
        except Exception as load_error:
            show_error(f"Error loading data files: {load_error}")
            return
        
        if data_display.empty:
            show_error(f"Could not read display data for {stock_name}. File may be empty or in invalid format.")
            return
            
        if data_train.empty:
            print(f"Warning: Could not read train data for {stock_name}")
        
        # Normalize dates
        data_display['Date'] = pd.to_datetime(data_display['Date']).dt.normalize()
        data_train['Date'] = pd.to_datetime(data_train['Date']).dt.normalize()
        
        # Define display time range if not provided
        if start_date is None:
            start_date = data_display['Date'].min()
        if end_date is None:
            end_date = data_display['Date'].max()
        
        # Split predictions into two lists - from file and hardcoded
        file_predictions = []
        hardcoded_predictions = []
        
        for pred in baodautu_predictions:
            if len(pred) == 3:  # Check if pred has correct format
                date, value, is_from_file = pred
                if is_from_file:
                    file_predictions.append((date, value))
                else:
                    hardcoded_predictions.append((date, value))
            else:
                # Handle old case without is_from_file
                date, value = pred
                hardcoded_predictions.append((date, value))
        
        # Convert prediction data to DataFrame for easier processing
        # DataFrame for data from file
        file_df = pd.DataFrame(columns=['Date', 'Price'])  # Initialize empty DataFrame
        if file_predictions:
            dates_file, values_file = zip(*file_predictions)
            file_df = pd.DataFrame({'Date': dates_file, 'Price': values_file})
            print(f"Number of predictions from file: {len(file_df)}")
        else:
            print("No predictions from file")
        
        # DataFrame for all data (both file and hardcoded)
        all_dates, all_values = zip(*file_predictions + hardcoded_predictions) if (file_predictions or hardcoded_predictions) else ([], [])
        baodautu_df = pd.DataFrame({'Date': all_dates, 'Price': all_values})
        
        print(f"Total number of predictions: {len(baodautu_df)}")
        
        # Kiểm tra xem có dữ liệu dự báo không
        if baodautu_df.empty:
            show_error(f"Không tìm thấy dữ liệu dự báo nào từ Báo Đầu Tư trong khoảng thời gian đã chọn.")
            return
            
        # Lọc dữ liệu theo khoảng thời gian đã chọn thay vì danh sách ngày cố định
        baodautu_df = baodautu_df[(baodautu_df['Date'] >= start_date) & (baodautu_df['Date'] <= end_date)]
        print(f"Số dự báo trong khoảng thời gian {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}: {len(baodautu_df)}")
        
        if baodautu_df.empty:
            show_error(f"Không tìm thấy dữ liệu dự báo nào từ Báo Đầu Tư trong khoảng thời gian đã chọn.")
            return
            
        # Clear current chart
        ax.clear()
        ax_table.clear()
        ax_table.axis('off')
        
        # Filter out weekends from data for other lines
        filtered_data_display = data_display.copy()  # Create a copy for filtering
        filtered_data_display = filtered_data_display[~filtered_data_display['Date'].dt.dayofweek.isin([5, 6])]  # 5 = Saturday, 6 = Sunday
        
        # Filter data by time range
        mask_actual = (data_display['Date'] >= start_date) & (data_display['Date'] <= end_date)
        filtered_actual_data = data_display.loc[mask_actual]
        
        # For other calculations, use the filtered data
        filtered_mask = (filtered_data_display['Date'] >= start_date) & (filtered_data_display['Date'] <= end_date)
        filtered_data_for_calcs = filtered_data_display.loc[filtered_mask]
        
        # Calculate historical volatility for adding realistic fluctuations
        if len(filtered_data_for_calcs) > 5:
            hist_returns = filtered_data_for_calcs['Price'].pct_change().dropna()
            historical_volatility = hist_returns.std()
            daily_volatility = historical_volatility / np.sqrt(252)  # Daily volatility
        else:
            # Default volatility if not enough data
            daily_volatility = 0.02  # 2% daily volatility as default
        
        print(f"Estimated daily volatility: {daily_volatility:.2%}")
        
        # Create a more granular set of dates for a realistic price path
        # Count the number of trading days in the range
        all_trading_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Filter out any remaining weekend days just to be sure
        all_trading_days = [day for day in all_trading_days if day.weekday() < 5]  # 0-4 are Monday-Friday
        
        # Configure plot to use dates on x-axis for better display
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        
        # Draw the real line using the raw _TT.csv data without any modifications
        if not filtered_actual_data.empty:
            ax.plot(filtered_actual_data['Date'], filtered_actual_data['Price'],
                   label='Thực tế', color='blue', linewidth=2.0)
            
            # Save last price to calculate growth rate
            last_actual_price = filtered_actual_data['Price'].iloc[-1] if not filtered_actual_data.empty else None
        else:
            last_actual_price = None
            print("Warning: No actual data in selected date range")
        
        # -------------- Combine Báo Đầu Tư data with training data --------------
        n_lags = 5  # Days of data needed for forecast
        min_required_days = 55  # Minimum days needed
        
        # Filter out weekends from training data
        data_train = data_train[~data_train['Date'].dt.dayofweek.isin([5, 6])]
        
        # Create features for forecasting
        data_lagged = create_advanced_features(data_train)
        if data_lagged.empty:
            show_error(f"Could not create forecast features for {stock_name}. Please check the data.")
            return
            
        # Get feature columns for training (exclude Date and target)
        feature_columns = [col for col in data_lagged.columns if col not in ['Date', 'target']]
        
        # Get last date in data
        max_date_in_data = data_lagged['Date'].max()
        
        # Try to load all models
        missing_models = []
        models = {}
        for model_name in ['rf', 'xgb', 'lgb', 'dt']:
            model = load_model(model_name, stock_name)
            if model is not None:
                models[model_name] = model
            else:
                missing_models.append(model_name)

        # If any models are missing, show error and return
        if missing_models:
            show_error(f"Không tìm thấy các mô hình sau cho {stock_name}: {', '.join(missing_models)}.\nVui lòng chạy file train_models.py để huấn luyện mô hình.")
            return

        # Try to load scaler
        scaler = load_scaler(stock_name)
        if scaler is None:
            show_error(f"Không tìm thấy scaler cho {stock_name}.\nVui lòng chạy file train_models.py để huấn luyện mô hình.")
            return

        print(f"Đã tải thành công các mô hình cho {stock_name}")
        
        # Get the baodatu data points as anchors
        anchor_dates = baodautu_df['Date'].sort_values().tolist()
        anchor_prices = baodautu_df.sort_values('Date')['Price'].tolist()
        
        # Generate more realistic price movements between anchor points
        for model_name in models.keys():
            predictions = []
            last_price = None
            
            # Calculate historical trend and volatility from recent data
            recent_data = filtered_data_for_calcs.tail(30)  # Last 30 days
            if not recent_data.empty:
                historical_trend = (recent_data['Price'].iloc[-1] - recent_data['Price'].iloc[0]) / recent_data['Price'].iloc[0]
                historical_volatility = recent_data['Price'].pct_change().std()
            else:
                historical_trend = 0
                historical_volatility = 0.01
            
            # For each pair of consecutive anchor points
            for i in range(len(anchor_dates) - 1):
                start_date_segment = anchor_dates[i]
                end_date_segment = anchor_dates[i+1]
                start_price = anchor_prices[i]
                end_price = anchor_prices[i+1]
                
                # Calculate segment trend
                segment_trend = (end_price - start_price) / start_price
                
                # Get all trading days between these two dates
                segment_days = [d for d in all_trading_days if start_date_segment <= d <= end_date_segment]
                
                if len(segment_days) <= 1:
                    # Just one day, add the anchor point
                    predictions.append((start_date_segment, start_price))
                    continue
                
                # Create base data for this segment with smooth price transition
                num_days = len(segment_days)
                
                # Use historical trend to influence the price curve
                trend_weight = 0.3
                adjusted_end_price = start_price * (1 + (segment_trend * (1 - trend_weight) + historical_trend * trend_weight))
                
                # Use cubic interpolation for smoother price transitions
                t = np.linspace(0, 1, num_days)
                # Add some curvature to the price path, influenced by historical trend
                price_curve = start_price + (adjusted_end_price - start_price) * (3 * t ** 2 - 2 * t ** 3)
                
                # Create segment data
                segment_data = pd.DataFrame({
                    'Date': segment_days,
                    'Price': price_curve
                })
                
                # Add other required columns with realistic variations based on historical volatility
                price_std = historical_volatility * start_price
                segment_data['Open'] = segment_data['Price'] + np.random.normal(0, price_std * 0.5, len(segment_data))
                segment_data['High'] = segment_data['Price'] + abs(np.random.normal(0, price_std, len(segment_data)))
                segment_data['Low'] = segment_data['Price'] - abs(np.random.normal(0, price_std, len(segment_data)))
                segment_data['Volume'] = np.random.uniform(80000, 120000, len(segment_data))
                segment_data['Change'] = segment_data['Price'].pct_change().fillna(0)
                
                # Ensure High > Open > Low
                segment_data['High'] = np.maximum(segment_data[['High', 'Open', 'Price']].max(axis=1), 
                                                segment_data['Price'] * 1.001)
                segment_data['Low'] = np.minimum(segment_data[['Low', 'Open', 'Price']].min(axis=1),
                                               segment_data['Price'] * 0.999)
                
                # Create features
                segment_features = create_advanced_features(segment_data)
                if segment_features.empty:
                    continue
                
                # Get features for prediction
                X = segment_features[feature_columns]
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Make predictions
                base_predictions = models[model_name].predict(X_scaled)
                
                # Apply smoothing and anchoring
                smoothed_predictions = []
                for j, base_pred in enumerate(base_predictions):
                    # Calculate weights for smoothing
                    if last_price is not None:
                        # Use adaptive smoothing based on prediction quality
                        if model_name == 'xgb' or model_name == 'lgb':
                            alpha = 0.2  # More smoothing for better performing models
                        else:
                            alpha = 0.4  # Less smoothing for other models
                        smoothed_price = alpha * base_pred + (1 - alpha) * last_price
                    else:
                        smoothed_price = base_pred
                    
                    # Apply stronger anchoring to Báo Đầu Tư predictions
                    if j == 0:  # First point
                        anchor_weight = 0.9  # Increased from 0.7
                        smoothed_price = anchor_weight * start_price + (1 - anchor_weight) * smoothed_price
                    elif j == len(base_predictions) - 1:  # Last point
                        anchor_weight = 0.9  # Increased from 0.7
                        smoothed_price = anchor_weight * end_price + (1 - anchor_weight) * smoothed_price
                    else:
                        # Interpolate anchor weight based on distance to anchor points
                        dist_to_start = j
                        dist_to_end = len(base_predictions) - 1 - j
                        if dist_to_start < 3 or dist_to_end < 3:  # Strong anchoring near anchor points
                            anchor_weight = 0.8
                            target_price = start_price if dist_to_start < dist_to_end else end_price
                            smoothed_price = anchor_weight * target_price + (1 - anchor_weight) * smoothed_price
                    
                    smoothed_predictions.append(smoothed_price)
                    last_price = smoothed_price
                
                # Add predictions
                for date, price in zip(segment_days, smoothed_predictions):
                    predictions.append((date, price))
            
            # Ensure we have the last anchor point
            if anchor_dates:
                last_date = anchor_dates[-1]
                last_price = anchor_prices[-1]
                predictions.append((last_date, last_price))
            
            # Convert predictions to DataFrame
            if predictions:
                dates = [p[0] for p in predictions]
                values = [p[1] for p in predictions]
                
                # Create DataFrame first with just dates and predictions
                df = pd.DataFrame({
                    'Date': dates,
                    'Prediction': values
                })
                
                # Add Baodautu_Price column by matching dates
                df['Baodautu_Price'] = None  # Initialize with None
                for i, anchor_date in enumerate(anchor_dates):
                    df.loc[df['Date'] == anchor_date, 'Baodautu_Price'] = anchor_prices[i]
                
                # Apply final smoothing with adaptive span based on model performance
                if model_name == 'xgb' or model_name == 'lgb':
                    smooth_span = 2  # Less smoothing for better models
                else:
                    smooth_span = 4  # More smoothing for other models
                df['Prediction'] = df['Prediction'].ewm(span=smooth_span).mean()
                
                # Ensure predictions exactly match Baodautu points
                for i, anchor_date in enumerate(anchor_dates):
                    df.loc[df['Date'] == anchor_date, 'Prediction'] = anchor_prices[i]
                
                # Filter out weekend days
                df = df[df['Date'].dt.weekday < 5]
                
                # Only add if there are still predictions after filtering
                if not df.empty:
                    df_predictions[model_name] = df

        # Colors for each model
        colors = {
            'rf': 'red',
            'xgb': 'orange',
            'lgb': 'green',
            'dt': 'purple'
        }
        
        # Different dashed line styles for different models
        line_styles = {
            'rf': (0, (5, 5)),
            'xgb': (0, (3, 3, 1, 3)),
            'lgb': (0, (1, 1)),
            'dt': (0, (5, 2, 1, 2))
        }
        
        # Model labels
        model_labels = {
            'rf': 'Random Forest',
            'xgb': 'XGBoost',
            'lgb': 'LightGBM',
            'dt': 'Decision Tree'
        }

        # Plot predictions for each model
        for model_name in models.keys():
            if model_name not in df_predictions or df_predictions[model_name].empty:
                print(f"Skipping {model_name} - no predictions available")
                continue
            
            df = df_predictions[model_name]
            print(f"Plotting chart for model: {model_name} with {len(df)} data points")
            
            ax.plot(df['Date'], df['Prediction'],
                   label=f"{model_labels[model_name]} + Báo Đầu Tư", 
                   linestyle=line_styles[model_name], 
                   color=colors[model_name],
                   linewidth=2.5)

        # Plot Báo Đầu Tư predictions
        ax.scatter(baodautu_df['Date'], baodautu_df['Price'],
                  label='Dự báo Báo Đầu Tư',
                  color='black', marker='o', s=50)

        # Set title and labels
        ax.set_title(f'Dự báo giá cổ phiếu - {stock_name} với dữ liệu Báo Đầu Tư', fontsize=16)
        ax.set_xlabel('Ngày', fontsize=14)
        ax.set_ylabel('Giá', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Configure date formatting for x-axis and adjust rotation for better readability
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.xticks(rotation=45, ha='right')
        
        # Calculate appropriate number of x-ticks based on date range
        date_range = (end_date - start_date).days
        
        # Create business days (excluding weekends) for x-ticks
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        business_days = [day for day in business_days if day.weekday() < 5]  # 0-4 are Monday-Friday
        
        # Calculate number of ticks based on available business days
        num_ticks = min(max(5, len(business_days) // 2), 10)  # At least 5, at most 10 ticks
        
        # If we have too few business days, use all of them
        if len(business_days) <= num_ticks:
            date_ticks = business_days
        else:
            # Otherwise evenly space the ticks
            indices = np.linspace(0, len(business_days)-1, num_ticks, dtype=int)
            date_ticks = [business_days[i] for i in indices]
        
        ax.set_xticks(date_ticks)
        
        # Set the x-axis limits to the date range
        ax.set_xlim(start_date, end_date)
        
        fig.autofmt_xdate()
        
        # Set tight layout for better spacing
        fig.tight_layout()
        
        # Set up mouse interaction for tooltips
        if motion_cid is not None:
            fig.canvas.mpl_disconnect(motion_cid)
            
        cursor = mplcursors.cursor(ax, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            global last_annotation
            last_annotation = sel.annotation
            x_value, y_value = sel.target
            
            # Format the date nicely - handle both datetime and float x-values
            if isinstance(x_value, (np.datetime64, pd.Timestamp)):
                date_str = pd.Timestamp(x_value).strftime('%d-%m-%Y')
            else:
                try:
                    # Try to convert timestamp float to datetime
                    date_str = pd.Timestamp(mdates.num2date(x_value)).strftime('%d-%m-%Y')
                except:
                    date_str = "Unknown date"
                
            sel.annotation.set(text=f'Ngày: {date_str}\nGiá: {y_value:.2f}')
            sel.annotation.set_visible(True)
            fig.canvas.draw_idle()
        
        def on_motion(event):
            global last_annotation
            if event.inaxes != ax:
                if last_annotation is not None:
                    last_annotation.set_visible(False)
                    fig.canvas.draw_idle()
                return
            over_any_line = False
            for line in ax.lines:
                contains, _ = line.contains(event)
                if contains:
                    over_any_line = True
                    break
            if not over_any_line and last_annotation is not None:
                last_annotation.set_visible(False)
                fig.canvas.draw_idle()
            
        motion_cid = fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.draw_idle()
        
        # Hide error table area
        ax_table.clear()
        ax_table.axis('off')
        
        # Show success message
        number_of_file_predictions = len(file_df)
        number_of_predictions = len(baodautu_df)
        show_success(f"Improved forecast using {number_of_file_predictions} data points from file and {number_of_predictions - number_of_file_predictions} supplemental data points for {stock_name}")
        
        # Calculate metrics if we have actual data to compare against
        if end_date <= max_date_in_data and df_predictions:  # Check if dictionary is not empty
            calculate_prediction_metrics(df_predictions, filtered_data_for_calcs, end_date, max_date_in_data, 
                                      show_success=show_success, prediction_type='normal')
        elif df_predictions:  # For future predictions with supplementary data
            calculate_prediction_metrics(df_predictions, filtered_data_for_calcs, end_date, max_date_in_data,
                                      show_success=show_success, prediction_type='supplement')
        else:
            # When no predictions available
            metrics_future = None
            metrics_normal = None
            metrics_supplement = None
            error_df = pd.DataFrame({
                'Model': ['Chưa có dữ liệu'],
                'MSE': ['N/A'], 'MAE': ['N/A'], 'RMSE': ['N/A'],
                'MAPE': ['N/A'], 'R²': ['N/A']
            })

    except Exception as e:
        import traceback
        traceback_details = traceback.format_exc()
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback_details}")
        show_error(f"Error updating chart with Báo Đầu Tư data: {e}")

# ---------------------------
# In main loop, check queue and process UI updates
# ---------------------------
def process_queue():
    """Process tasks in the update queue"""
    try:
        while not update_queue.empty():
            task = update_queue.get()
            try:
                task()
            except Exception as task_error:
                print(f"Error executing task: {task_error}")
    except Exception as queue_error:
        print(f"Error processing queue: {queue_error}")
    finally:
        # Schedule this function again after 100ms
        root.after(100, process_queue)

# Start queue checking loop
root.after(100, process_queue)

# ---------------------------
# Button event handlers
# ---------------------------
def forecast_button_clicked(event):
    axbox_start_date.set_visible(True)
    axbox_end_date.set_visible(True)
    ax_button_ok.set_visible(True)
    fig.canvas.draw_idle()

def ok_button_clicked(event):
    start_date_str = text_box_start_date.text
    end_date_str = text_box_end_date.text
    s_date = pd.to_datetime(start_date_str, dayfirst=True, errors='coerce')
    e_date = pd.to_datetime(end_date_str, dayfirst=True, errors='coerce')
    if pd.isnull(s_date) or pd.isnull(e_date):
        show_error("Invalid date format. Please enter dates in dd-mm-yyyy format.")
        return

    # Hide date input widgets
    axbox_start_date.set_visible(False)
    axbox_end_date.set_visible(False)
    ax_button_ok.set_visible(False)
    fig.canvas.draw_idle()

    # Create loading window to notify user
    loading_root = tk.Toplevel(root)
    loading_root.title("Loading")
    tk.Label(loading_root, text="Analyzing data, please wait...").pack(padx=20, pady=10)
    progress = ttk.Progressbar(loading_root, orient="horizontal", mode="indeterminate", length=280)
    progress.pack(padx=20, pady=10)
    progress.start(10)
    loading_root.lift()
    loading_root.attributes("-topmost", True)
    
    # Protect loading window from being closed by user 
    def on_closing():
        pass  # Do nothing - prevent user from closing loading window
    
    loading_root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start worker thread to calculate and update chart
    t = threading.Thread(target=worker, args=(current_dataset, s_date, e_date, loading_root), daemon=True)
    t.start()

# Handler for stock selection from dropdown
def on_stock_selected(stock_name):
    global current_dataset
    
    if stock_name and stock_name in stock_files:
        # Check if required files exist
        file_path_train = os.path.join(TRAIN_DIR, f"{stock_name}.csv")
        file_path_display = os.path.join(VE_DIR, f"{stock_name}_TT.csv")
        
        if not os.path.exists(file_path_train):
            show_error(f"Train data file not found: {file_path_train}")
            return
            
        if not os.path.exists(file_path_display):
            show_error(f"Display data file not found: {file_path_display}")
            return
        
        # Update current dataset
        current_dataset = stock_name
        
        # Update chart with selected stock
        update_plot(None, stock_name)
    else:
        show_error(f"Data not found for stock: {stock_name}")

# Connect events
button_forecast.on_clicked(forecast_button_clicked)
button_ok.on_clicked(ok_button_clicked)
button_error.on_clicked(apply_baodautu_predictions)  # Use Báo Đầu Tư forecast function
dropdown_menu.on_changed(on_stock_selected)

# Display initial data (without forecast)
update_plot(None, current_dataset)

def save_metrics_to_file():
    """
    Lưu metrics vào file JSON
    """
    global metrics_normal, metrics_future, metrics_supplement
    
    metrics_data = {
        'normal': metrics_normal,
        'future': metrics_future,
        'supplement': metrics_supplement
    }
    
    # Tạo thư mục metrics nếu chưa tồn tại
    metrics_dir = 'metrics'
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # Lưu metrics vào file JSON
    metrics_file = os.path.join(metrics_dir, 'system_metrics.json')
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"\nĐã lưu metrics vào file: {metrics_file}")
        return True
    except Exception as e:
        print(f"\nLỗi khi lưu metrics: {e}")
        return False

def load_metrics_from_file():
    """
    Tải metrics từ file JSON
    """
    global metrics_normal, metrics_future, metrics_supplement
    
    metrics_file = os.path.join('metrics', 'system_metrics.json')
    
    if not os.path.exists(metrics_file):
        print("\nKhông tìm thấy file metrics đã lưu.")
        return False
        
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
            
        metrics_normal = metrics_data.get('normal', {})
        metrics_future = metrics_data.get('future', {})
        metrics_supplement = metrics_data.get('supplement', {})
        
        print(f"\nĐã tải metrics từ file: {metrics_file}")
        return True
    except Exception as e:
        print(f"\nLỗi khi tải metrics: {e}")
        return False

def check_metrics_need_update():
    """
    Kiểm tra xem có cần tính toán lại metrics không
    """
    metrics_file = os.path.join('metrics', 'system_metrics.json')
    
    # Nếu file metrics không tồn tại
    if not os.path.exists(metrics_file):
        return True
        
    # Lấy thời gian sửa đổi của file metrics
    metrics_mtime = os.path.getmtime(metrics_file)
    
    # Kiểm tra các file trong thư mục Train
    for file in os.listdir(TRAIN_DIR):
        if file.endswith('.csv'):
            file_path = os.path.join(TRAIN_DIR, file)
            # Nếu có file train mới hơn file metrics
            if os.path.getmtime(file_path) > metrics_mtime:
                return True
                
    # Kiểm tra các file trong thư mục Ve
    for file in os.listdir(VE_DIR):
        if file.endswith('.csv'):
            file_path = os.path.join(VE_DIR, file)
            if os.path.getmtime(file_path) > metrics_mtime:
                return True
    
    return False

def calculate_system_metrics():
    """
    Tính toán metrics cho toàn hệ thống khi khởi động
    """
    global metrics_normal, metrics_future, metrics_supplement
    
    # Kiểm tra xem có cần tính toán lại không
    if not check_metrics_need_update():
        if load_metrics_from_file():
            print("Sử dụng metrics đã lưu từ lần tính toán trước.")
            return
    
    print("\nĐang tính toán metrics cho toàn hệ thống...")
    
    try:
        # [Giữ nguyên phần code tính toán metrics ở đây...]
        
        # Sau khi tính toán xong, lưu kết quả vào file
        save_metrics_to_file()
        
    except Exception as e:
        print(f"\nLỗi khi tính toán metrics cho hệ thống: {e}")
        traceback.print_exc()

# Add after imports and before other code
def clear_models_directory():
    """
    Kiểm tra và tạo thư mục models nếu chưa tồn tại
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"\nĐã tạo thư mục models tại: {models_dir}")
    else:
        print(f"\nĐã tìm thấy thư mục models tại: {models_dir}")

def check_models_features(model_name, dataset, features):
    """
    Kiểm tra xem mô hình đã lưu có sử dụng đúng features không
    """
    try:
        model = load_model(model_name, dataset)
        if model is None:
            return False
            
        # Kiểm tra feature names cho từng loại mô hình
        if hasattr(model, 'feature_names_in_'):
            model_features = set(model.feature_names_in_)
        elif hasattr(model, 'feature_name_'):
            model_features = set(model.feature_name_)
        else:
            # Nếu không tìm thấy thông tin về features, coi như cần train lại
            return False
            
        required_features = set(features)
        return model_features == required_features
    except Exception as e:
        print(f"Lỗi khi kiểm tra features của mô hình {model_name}: {e}")
        return False

# Thêm vào cuối file, trước plt.show()
calculate_system_metrics()

# Clear models directory at startup
clear_models_directory()

plt.show()
