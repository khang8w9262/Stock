# Stock Price Prediction App

This is a Streamlit web application for predicting stock prices using various machine learning models.

## Setup

### Install Dependencies

```bash
pip install -r streamlit_requirements.txt
```

### Run the Streamlit App

```bash
streamlit run UI.py
```

## Features

- Interactive stock price visualization
- Multiple prediction models (Random Forest, XGBoost, LightGBM, Decision Tree)
- Historical and future price forecasting
- Supplemental forecasting with Baodautu data
- Error metrics analysis
- Downloadable metrics reports

## File Structure

- `UI.py`: Main Streamlit application
- `back.py`: Backend functions for loading data and making predictions
- `Train/`: Directory containing historical stock data CSV files
- `Ve/`: Directory containing display data for visualization
- `models/`: Directory containing trained ML models
- `metrics/`: Directory containing error metrics
- `Baodautu/`: Directory containing supplemental prediction data

## Usage

1. Select a stock from the dropdown menu in the sidebar
2. Set the date range for analysis
3. Click "Forecast" to generate predictions
4. Click "Supplement" to include Baodautu predictions
5. Click "View Metrics" to see performance metrics

## Notes

- Make sure all required directories (`Train/`, `Ve/`, `models/`, `metrics/`, `Baodautu/`) exist before running the app
- Stock data files should follow the naming convention `<stock_symbol>.csv` in the `Train/` directory
- Display data files should follow the naming convention `<stock_symbol>_TT.csv` in the `Ve/` directory
