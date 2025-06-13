import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Optional, Dict, List, Tuple, Any
import traceback
import time

# Import backend functions
from back import (
    load_data,
    create_advanced_features,
    simulate_future_price,
    load_model,
    load_scaler,
    load_feature_order,
    find_baodautu_prediction
)

# Configure directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
VE_DIR = os.path.join(BASE_DIR, 'Ve')
BAODAUTU_DIR = os.path.join(BASE_DIR, 'Baodautu')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
METRICS_FILE = os.path.join(METRICS_DIR, 'metrics.json')

# Create required directories
for directory in [TRAIN_DIR, VE_DIR, BAODAUTU_DIR, MODELS_DIR, METRICS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide"
)

# Initialize session state
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'error_df' not in st.session_state:
    st.session_state.error_df = None
if 'metrics_normal' not in st.session_state:
    st.session_state.metrics_normal = None
if 'metrics_future' not in st.session_state:
    st.session_state.metrics_future = None
if 'metrics_supplement' not in st.session_state:
    st.session_state.metrics_supplement = None
if 'last_plot' not in st.session_state:
    st.session_state.last_plot = None
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Initialize session state variables for controls
if 'forecast_btn' not in st.session_state:
    st.session_state.forecast_btn = False
if 'supplement_btn' not in st.session_state:
    st.session_state.supplement_btn = False
if 'metrics_btn' not in st.session_state:
    st.session_state.metrics_btn = False

# Initialize date range with reasonable defaults
today = datetime.now()
if 'start_date' not in st.session_state:
    st.session_state.start_date = today - pd.Timedelta(days=30)  # Default to last 30 days
if 'end_date' not in st.session_state:
    st.session_state.end_date = today + pd.Timedelta(days=10)    # Default to 10 days in future

# Initialize stock list
if 'stock_files' not in st.session_state:
    if os.path.exists(TRAIN_DIR):
        available_files = [f.split('.')[0] for f in os.listdir(TRAIN_DIR) if f.endswith('.csv')]
        if available_files:
            st.session_state.stock_files = sorted(available_files)
        else:
            st.session_state.stock_files = ["AAPL", "MSFT", "GOOGL"]
    else:
        st.session_state.stock_files = ["AAPL", "MSFT", "GOOGL"]

# ---------------------------
# Utility functions
# ---------------------------
def show_error(message: str) -> None:
    """Display error message"""
    st.error(message)

def show_warning(message: str) -> None:
    """Display warning message"""
    st.warning(message)
    
def show_info(message: str) -> None:
    """Display info message"""
    st.info(message)
    
def show_success(message: str) -> None:
    """Display success message"""
    st.success(message)

def get_predictions(model, df: pd.DataFrame, dataset: str) -> Optional[np.ndarray]:
    """Get predictions from a model for given data"""
    try:
        # Load scaler and feature order
        scaler = load_scaler(dataset)
        feature_order = load_feature_order(dataset)
        
        if scaler is None or feature_order is None:
            show_warning("Could not load scaler or feature order")
            return None
            
        # Create features
        features_df = create_advanced_features(df)
        if features_df.empty:
            show_warning("Could not create features")
            return None
            
        # Select and scale features
        X = features_df[feature_order]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        return model.predict(X_scaled)
        
    except Exception as e:
        show_error(f"Error getting predictions: {str(e)}")
        return None

def update_chart(
    fig: go.Figure,
    title: str,
    x_title: str = "Date",
    y_title: str = "Price"
) -> None:
    """Update chart layout and display"""
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Initialize state and data
# ---------------------------
def initialize_stock_files():
    """Initialize list of available stock files"""
    if os.path.exists(TRAIN_DIR):
        try:
            available_files = os.listdir(TRAIN_DIR)
            stock_files = [f.split('.')[0] for f in available_files if f.endswith('.csv')]
            if not stock_files:
                st.warning(f"No CSV files found in directory {TRAIN_DIR}")
                stock_files = ["AAPL", "MSFT", "GOOGL"]
                st.info(f"Using sample stock list")
        except Exception as e:
            st.error(f"Error reading directory {TRAIN_DIR}: {str(e)}")
            stock_files = ["AAPL", "MSFT", "GOOGL"]
    else:
        st.warning(f"Directory {TRAIN_DIR} does not exist!")
        os.makedirs(TRAIN_DIR, exist_ok=True)
        st.info(f"Created Train directory at: {TRAIN_DIR}")
        stock_files = ["AAPL", "MSFT", "GOOGL"]
    
    return sorted(stock_files)

def display_error_table(error_df: pd.DataFrame):
    """Display error metrics in a Streamlit table"""
    if error_df is None:
        st.error("No error data available. Please run the forecast first.")
        return
        
    df_error_display = error_df.copy()
    df_error_display.iloc[:, 1:] = df_error_display.iloc[:, 1:].applymap(lambda x: f"{x:.3f}")
    st.dataframe(df_error_display, use_container_width=True)

# ---------------------------
# Initialize Figure and Axes for matplotlib
# ---------------------------
fig = go.Figure()

# ---------------------------
# Function to display success message
# ---------------------------
def show_success(message):
    st.success(message)

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
#dropdown_menu = DropdownMenu(ax_dropdown, stock_files, initial=0, label="Stock: ", width=0.15, height=0.04)

# Fixed positions for buttons in the top right (above the plot)
# "Forecast" button
#ax_button_forecast = plt.axes([0.78, 0.92, 0.10, 0.05])  # Move above plot area
#button_forecast = Button(ax_button_forecast, 'Forecast')

# "Supplement" button 
#ax_button_error = plt.axes([0.90, 0.92, 0.10, 0.05])  # Move above plot area
#button_error = Button(ax_button_error, 'Supplement')

# "Start date" and "End date" input boxes with "OK" button (middle bottom)
# These will be shown/hidden but position remains fixed
#axbox_start_date = plt.axes([0.30, 0.10, 0.15, 0.05])  # Lower position
#axbox_end_date   = plt.axes([0.60, 0.10, 0.15, 0.05])  # Lower position
#text_box_start_date = TextBox(axbox_start_date, 'Start date:', initial='13-03-2025')
#text_box_end_date   = TextBox(axbox_end_date, 'End date:', initial='26-03-2025')

#ax_button_ok = plt.axes([0.45, 0.02, 0.1, 0.05])  # Lower position
#button_ok = Button(ax_button_ok, 'OK')

# Initially hide date input widgets
#axbox_start_date.set_visible(False)
#axbox_end_date.set_visible(False)
#ax_button_ok.set_visible(False)

# ---------------------------
# Metrics handling functions
# ---------------------------
def store_future_metrics(metrics_dict):
    """
    Store metrics for future predictions
    """
    st.session_state.metrics_future = metrics_dict.copy()

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

def load_metrics_from_file():
    """Load metrics from the metrics file"""
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        show_error(f"Error loading metrics: {str(e)}")
        return {}

def save_metrics_to_file(metrics_data):
    """Save metrics to the metrics file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        
        # Load existing data
        existing_data = {}
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                existing_data = json.load(f)
                
        # Update with new data
        existing_data.update(metrics_data)
        
        # Save back to file
        with open(METRICS_FILE, 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        show_success(f"Metrics saved to {METRICS_FILE}")
    except Exception as e:
        show_error(f"Error saving metrics: {str(e)}")

def show_metrics_comparison():
    """Display metrics comparison table"""
    try:
        metrics_data = load_metrics_from_file()
        if not metrics_data:
            st.warning("No metrics data available")
            return
            
        # Create tabs for different prediction types
        tab_names = {
            'normal': 'Normal Forecast',
            'future': 'Future Forecast',
            'supplement': 'Supplemental Forecast'
        }
        
        tabs = st.tabs(list(tab_names.values()))
        
        for tab, (pred_type, tab_name) in zip(tabs, tab_names.items()):
            with tab:
                if pred_type in metrics_data and metrics_data[pred_type]:
                    metrics_df = pd.DataFrame(metrics_data[pred_type]).T
                    
                    # Format metrics
                    formatted_df = metrics_df.applymap(lambda x: f"{x:.6f}")
                    
                    # Display metrics table
                    st.write(f"### {tab_name} Metrics")
                    st.dataframe(formatted_df)
                    
                    # Add download button for metrics
                    csv = formatted_df.to_csv()
                    st.download_button(
                        label=f"Download {tab_name} Metrics",
                        data=csv,
                        file_name=f"{pred_type}_metrics.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No metrics available for {tab_name}")
                    
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def show_header():
    """Display app header and controls"""
    st.title("Stock Price Prediction")
    st.markdown("---")

def show_sidebar_controls():
    """Display sidebar controls"""
    with st.sidebar:
        st.title("Controls")
        
        # Stock selection
        selected_stock = st.selectbox(
            "Select Stock:",
            options=st.session_state.stock_files,
            index=st.session_state.stock_files.index(st.session_state.current_dataset) if st.session_state.current_dataset in st.session_state.stock_files else 0
        )
        
        # Date range
        st.subheader("Date Range")
        start_date = st.date_input("Start Date", value=st.session_state.start_date)
        end_date = st.date_input("End Date", value=st.session_state.end_date)
        
        # Action buttons
        st.subheader("Actions")
        col1, col2 = st.columns(2)
        with col1:
            forecast = st.button("Forecast", use_container_width=True)
        with col2:
            supplement = st.button("Supplement", use_container_width=True)
            
        metrics = st.button("View Metrics", use_container_width=True)
        
        return selected_stock, start_date, end_date, forecast, supplement, metrics

def process_stock_data(stock_name):
    """Load and process stock data"""
    try:
        file_path_train = os.path.join(TRAIN_DIR, f"{stock_name}.csv")
        file_path_display = os.path.join(VE_DIR, f"{stock_name}_TT.csv")
        
        if not os.path.exists(file_path_train):
            st.error(f"Train data file not found: {file_path_train}")
            return None, None
            
        if not os.path.exists(file_path_display):
            st.error(f"Display data file not found: {file_path_display}")
            return None, None
            
        # Load data
        df_train = pd.read_csv(file_path_train, encoding='utf-8')
        df_display = pd.read_csv(file_path_display, encoding='utf-8')
        
        # Convert dates
        df_train['Date'] = pd.to_datetime(df_train['Ngày'], format='%d/%m/%Y')
        df_display['Date'] = pd.to_datetime(df_display['Ngày'], format='%d/%m/%Y')
        
        return df_train, df_display
        
    except Exception as e:
        st.error(f"Error processing stock data: {str(e)}")
        return None, None

def get_stock_predictions(stock_name, df):
    """Get predictions from all models for a stock"""
    predictions = {}
    
    # Load models
    models = load_models(stock_name)
    if not models:
        st.warning(f"No models available for {stock_name}")
        return None
        
    # Load scaler and feature order
    scaler = load_scaler(stock_name)
    feature_order = load_feature_order(stock_name)
    
    if scaler is None or feature_order is None:
        st.warning(f"Missing scaler or feature order for {stock_name}")
        return None
        
    # Create features
    features_df = create_advanced_features(df)
    if features_df.empty:
        st.warning("Could not create features")
        return None
        
    # Make predictions with each model
    for model_name, model in models.items():
        try:
            X = features_df[feature_order]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predictions[model_name] = pred
        except Exception as e:
            st.warning(f"Error getting predictions from {model_name} model: {str(e)}")
            continue
            
    return predictions

def load_models(stock_name):
    """Load all models for a stock"""
    models = {}
    model_types = ['rf', 'xgb', 'lgb', 'dt']
    
    for model_type in model_types:
        try:
            model = load_model(model_type, stock_name)
            if model is not None:
                models[model_type] = model
        except Exception as e:
            st.warning(f"Error loading {model_type} model: {str(e)}")
            continue
            
    return models if models else None

def initialize_app():
    """Initialize app state and load stock files"""
    # Load available stock files
    if os.path.exists(TRAIN_DIR):
        available_files = [f.split('.')[0] for f in os.listdir(TRAIN_DIR) if f.endswith('.csv')]
        if available_files:
            st.session_state.stock_files = sorted(available_files)
        else:
            st.session_state.stock_files = ["AAPL", "MSFT", "GOOGL"]  # Sample stocks
    else:
        st.session_state.stock_files = ["AAPL", "MSFT", "GOOGL"]  # Sample stocks

# ---------------------------
# Event handlers and UI logic
# ---------------------------
def handle_forecast():
    """Handle forecast button click"""
    if not st.session_state.forecast_btn:
        return
    
    try:
        st.session_state.is_processing = True
        with st.spinner("Generating forecast..."):
            selected_stock = st.session_state.selected_stock
            start_date = pd.to_datetime(st.session_state.start_date)
            end_date = pd.to_datetime(st.session_state.end_date)
            
            # Update the plot
            fig = update_plot(selected_stock, start_date, end_date, show_predictions=True)
            st.session_state.current_dataset = selected_stock
            st.session_state.last_plot = fig
            
            # Reset button state
            st.session_state.forecast_btn = False
    except Exception as e:
        show_error(f"Error generating forecast: {str(e)}")
    finally:
        st.session_state.is_processing = False

def handle_supplement():
    """Handle supplement button click"""
    if not st.session_state.supplement_btn:
        return
    
    try:
        st.session_state.is_processing = True
        with st.spinner("Generating supplemental forecast..."):
            selected_stock = st.session_state.selected_stock
            start_date = pd.to_datetime(st.session_state.start_date)
            end_date = pd.to_datetime(st.session_state.end_date)
            
            # Get Baodautu predictions
            baodautu_predictions = []
            current_date = start_date
            while current_date <= end_date:
                price, found = find_baodautu_prediction(selected_stock, current_date)
                if found:
                    baodautu_predictions.append((current_date, price))
                current_date += pd.Timedelta(days=1)
            
            if not baodautu_predictions:
                show_warning(f"No Baodautu predictions found for {selected_stock}. Showing regular forecast.")
                fig = update_plot(selected_stock, start_date, end_date, show_predictions=True)
            else:
                fig = update_plot_with_baodautu(selected_stock, start_date, end_date, baodautu_predictions)
            
            st.session_state.current_dataset = selected_stock
            st.session_state.last_plot = fig
            
            # Reset button state
            st.session_state.supplement_btn = False
    except Exception as e:
        show_error(f"Error generating supplemental forecast: {str(e)}")
    finally:
        st.session_state.is_processing = False

def handle_metrics():
    """Handle metrics button click"""
    if not st.session_state.metrics_btn:
        return
        
    try:
        st.session_state.is_processing = True
        show_metrics_comparison()
        # Reset button state
        st.session_state.metrics_btn = False
    except Exception as e:
        show_error(f"Error showing metrics: {str(e)}")
    finally:
        st.session_state.is_processing = False

def update_plot(stock_name: str, start_date=None, end_date=None, show_predictions=False) -> Optional[go.Figure]:
    """Update the plot with new data and predictions"""
    try:
        print(f"Updating plot for {stock_name} from {start_date} to {end_date}")
        
        # Build file paths and read data
        file_path_train = os.path.join(TRAIN_DIR, f"{stock_name}.csv")
        file_path_display = os.path.join(VE_DIR, f"{stock_name}_TT.csv")
        
        if not os.path.exists(file_path_train) or not os.path.exists(file_path_display):
            show_error(f"Data files not found for {stock_name}")
            return None
            
        df_train_full = pd.read_csv(file_path_train, encoding='utf-8', sep=',')
        df_display_full = pd.read_csv(file_path_display, encoding='utf-8', sep=',')
        
        # Convert dates and sort
        df_train_full['Date'] = pd.to_datetime(df_train_full['Ngày'], format='%d/%m/%Y')
        df_display_full['Date'] = pd.to_datetime(df_display_full['Ngày'], format='%d/%m/%Y')
        df_train_full = df_train_full.sort_values('Date').reset_index(drop=True)
        df_display_full = df_display_full.sort_values('Date').reset_index(drop=True)

        last_historical_date = df_train_full['Date'].max()
        
        # Create new figure
        fig = go.Figure()
        
        # Plot actual historical data within the selected range
        df_display_hist = df_display_full[
            (df_display_full['Date'] >= start_date) & (df_display_full['Date'] <= end_date)
        ]
        
        if not df_display_hist.empty:
            fig.add_trace(go.Scatter(
                x=df_display_hist['Date'], y=df_display_hist['Lần cuối'],
                name='Actual Price', mode='lines', line=dict(color='blue', width=2),
                hoverinfo='x+y+name'
            ))
        
        if show_predictions:
            models = load_model(stock_name)
            colors = {'rf': 'red', 'xgb': 'orange', 'lgb': 'green', 'dt': 'purple'}
            dash_styles = {'rf': 'dash', 'xgb': 'dashdot', 'lgb': 'dot', 'dt': 'dash'}
            
            # Future predictions
            if end_date > last_historical_date:
                forecast_start_date = max(start_date, last_historical_date + pd.Timedelta(days=1))
                future_dates = pd.bdate_range(start=forecast_start_date, end=end_date)
                
                if not future_dates.empty and models:
                    scaler = load_scaler(stock_name)
                    
                    # Map column names for feature creation
                    df_train_mapped = df_train_full.copy()
                    df_train_mapped.rename(columns={
                        'Lần cuối': 'Price', 'Mở': 'Open', 'Cao': 'High',
                        'Thấp': 'Low', 'KL': 'Volume', '% Thay đổi': 'Change'
                    }, inplace=True)
                    
                    all_features = create_advanced_features(df_train_mapped)
                    
                    if all_features is not None and not all_features.empty:
                        last_feature_vector = all_features.iloc[[-1]]
                        
                        for model_name, model in models.items():
                            future_predictions = simulate_future_price(
                                model, last_feature_vector, scaler, future_dates, stock_name=stock_name
                            )
                            
                            if future_predictions:
                                pred_dates, pred_values = zip(*future_predictions)
                                fig.add_trace(go.Scatter(
                                    x=list(pred_dates), y=list(pred_values),
                                    name=f'Forecast (Future) {model_name.upper()}',
                                    mode='lines', 
                                    line=dict(color=colors.get(model_name, 'gray'), dash=dash_styles.get(model_name, 'solid'))
                                ))
            
            # Historical predictions
            historical_prediction_data = df_train_full[
                (df_train_full['Date'] >= start_date) & 
                (df_train_full['Date'] <= min(end_date, last_historical_date))
            ]
            
            if not historical_prediction_data.empty:
                for model_name, model in models.items():
                    predictions = get_predictions(model, historical_prediction_data.copy(), stock_name)
                    
                    if predictions is not None and len(predictions) > 0:
                        # Align predictions with dates
                        prediction_dates = historical_prediction_data['Date'].iloc[-len(predictions):]
                        
                        fig.add_trace(go.Scatter(
                            x=prediction_dates, y=predictions,
                            name=f'Forecast ({model_name.upper()})',
                            mode='lines', 
                            line=dict(color=colors.get(model_name, 'gray'), dash=dash_styles.get(model_name, 'solid'))
                        ))
                
                # Calculate and store error metrics if predictions are shown
                metrics = {}
                for model_name in models:
                    historical_errors = calculate_model_errors(stock_name, model_name)
                    if historical_errors:
                        metrics[model_name] = historical_errors
                
                if metrics:
                    st.session_state.metrics_normal = metrics
                    
                    # Create error dataframe for display
                    error_data = {'Stock': [stock_name]}
                    for model_name, model_metrics in metrics.items():
                        for metric_name, metric_value in model_metrics.items():
                            if metric_name not in error_data:
                                error_data[f"{metric_name}"] = []
                            error_data[f"{metric_name}"].append(metric_value)
                    
                    st.session_state.error_df = pd.DataFrame(error_data)
        
        # Update chart layout
        update_chart(fig, f"Stock Price Prediction - {stock_name}")
        return fig
        
    except Exception as e:
        show_error(f"Error updating plot: {str(e)}")
        st.error(traceback.format_exc())
        return None

def update_plot_with_baodautu(stock_name: str, start_date, end_date, baodautu_predictions) -> Optional[go.Figure]:
    """Update plot with Baodautu predictions anchoring"""
    try:
        # Build file paths and read data
        file_path_train = os.path.join(TRAIN_DIR, f"{stock_name}.csv")
        file_path_display = os.path.join(VE_DIR, f"{stock_name}_TT.csv")
        
        if not os.path.exists(file_path_train) or not os.path.exists(file_path_display):
            show_error(f"Data files not found for {stock_name}")
            return None
            
        df_train_full = pd.read_csv(file_path_train, encoding='utf-8', sep=',')
        df_display_full = pd.read_csv(file_path_display, encoding='utf-8', sep=',')
        
        # Convert dates and sort
        df_train_full['Date'] = pd.to_datetime(df_train_full['Ngày'], format='%d/%m/%Y')
        df_display_full['Date'] = pd.to_datetime(df_display_full['Ngày'], format='%d/%m/%Y')
        df_train_full = df_train_full.sort_values('Date').reset_index(drop=True)
        df_display_full = df_display_full.sort_values('Date').reset_index(drop=True)
        
        # Create new figure
        fig = go.Figure()
        
        # Plot actual historical data
        df_display_hist = df_display_full[
            (df_display_full['Date'] >= start_date) & (df_display_full['Date'] <= end_date)
        ]
        
        if not df_display_hist.empty:
            fig.add_trace(go.Scatter(
                x=df_display_hist['Date'], y=df_display_hist['Lần cuối'],
                name='Actual Price', mode='lines', line=dict(color='blue', width=2),
                hoverinfo='x+y+name'
            ))
        
        # Load models and support tools
        models = load_model(stock_name)
        scaler = load_scaler(stock_name)
        feature_order = load_feature_order(stock_name)
        
        if not models or not scaler or not feature_order:
            show_error(f"Could not load models or scaler for {stock_name}")
            return None
        
        # Prepare for simulation
        last_historical_date = df_train_full['Date'].max()
        last_price = df_train_full.iloc[-1]['Lần cuối']
        
        # Map column names for feature creation
        df_train_mapped = df_train_full.copy()
        df_train_mapped.rename(columns={
            'Lần cuối': 'Price', 'Mở': 'Open', 'Cao': 'High',
            'Thấp': 'Low', 'KL': 'Volume', '% Thay đổi': 'Change'
        }, inplace=True)
        
        # Parse Volume column
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
        
        # Main loop: for each model, generate the anchored prediction line
        for model_name, model in models.items():
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
                    name=f'Forecast ({model_name.upper()})',
                    mode='lines', line=dict(color=colors.get(model_name, 'gray'), dash=dash_styles.get(model_name, 'solid'))
                ))
        
        # Plot the anchor markers
        bd_dates, bd_prices = zip(*baodautu_predictions)
        fig.add_trace(go.Scatter(
            x=bd_dates, y=bd_prices, name='Baodautu Forecast', mode='markers',
            marker=dict(color='cyan', size=10, symbol='star')
        ))
        
        # Record metrics for this supplemental prediction
        metrics = {}
        for model_name in models:
            historical_errors = calculate_model_errors(stock_name, model_name)
            if historical_errors:
                metrics[model_name] = historical_errors
        
        if metrics:
            st.session_state.metrics_supplement = metrics
        
        # Update chart layout
        update_chart(fig, f"Stock Price Prediction with Baodautu - {stock_name}")
        return fig
        
    except Exception as e:
        show_error(f"Error updating plot with Baodautu: {str(e)}")
        st.error(traceback.format_exc())
        return None

def calculate_model_errors(stock_name, model_name):
    """Calculate error metrics for a model on historical data"""
    try:
        # Load historical data
        file_path_train = os.path.join(TRAIN_DIR, f"{stock_name}.csv")
        if not os.path.exists(file_path_train):
            return None
            
        df = pd.read_csv(file_path_train, encoding='utf-8', sep=',')
        df['Date'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')
        
        # Only use last 30 days for metrics
        df = df.sort_values('Date').tail(30).reset_index(drop=True)
        
        # Load model and make predictions
        model = load_model(stock_name).get(model_name)
        if model is None:
            return None
            
        predictions = get_predictions(model, df.copy(), stock_name)
        if predictions is None or len(predictions) == 0:
            return None
            
        # Calculate metrics on the available predictions
        actual = df['Lần cuối'].iloc[-len(predictions):].values
        pred = predictions
        
        # Calculate error metrics
        metrics = {
            'MAE': np.mean(np.abs(actual - pred)),
            'MSE': np.mean((actual - pred)**2),
            'RMSE': np.sqrt(np.mean((actual - pred)**2)),
            'MAPE': np.mean(np.abs((actual - pred) / actual)) * 100
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

# ---------------------------
# Main application function
# ---------------------------
def main():
    """Main application function"""
    # Update header
    show_header()
    
    # Show sidebar controls
    selected_stock, start_date, end_date, forecast_clicked, supplement_clicked, metrics_clicked = show_sidebar_controls()
    
    # Handle button clicks
    if forecast_clicked:
        handle_forecast()
    elif supplement_clicked:
        handle_supplement()
    elif metrics_clicked:
        handle_metrics()
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Chart", "Error Analysis"])
    
    with tab1:
        # Chart area
        if st.session_state.last_plot is not None:
            st.plotly_chart(st.session_state.last_plot, use_container_width=True)
        elif st.session_state.current_dataset is not None:
            # Show initial chart without predictions
            fig = update_plot(st.session_state.current_dataset,
                            pd.to_datetime(start_date),
                            pd.to_datetime(end_date),
                            show_predictions=False)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Error analysis area
        if st.session_state.error_df is not None:
            display_error_table(st.session_state.error_df)
            
    # Show loading indicator while processing
    if st.session_state.is_processing:
        st.spinner("Processing...")

if __name__ == "__main__":
    # Initialize state if not already initialized
    if 'stock_files' not in st.session_state:
        st.session_state.stock_files = initialize_stock_files()
    if 'last_plot' not in st.session_state:
        st.session_state.last_plot = None
        
    # Run the main app
    main()
