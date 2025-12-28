# Global Stock Price Prediction Application

## Overview

This is a comprehensive stock price prediction application built with Streamlit that provides real-time stock analysis and machine learning-based price forecasting. The application integrates financial data fetching, technical analysis, interactive charting, and predictive modeling to offer users insights into stock market trends and future price movements. It supports global stock markets and provides both visual analytics and numerical predictions with accuracy metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework for rapid prototyping and deployment
- **Layout**: Wide layout with expandable sidebar for user controls and parameter selection
- **Styling**: Custom CSS implementation for dark theme with branded color scheme (#00d4aa accent color)
- **Interactivity**: Real-time parameter adjustment and dynamic chart updates based on user selections

### Backend Architecture
- **Modular Design**: Four-component architecture with clear separation of concerns:
  - `app.py`: Main application orchestrator and UI controller
  - `data_fetcher.py`: Financial data acquisition and preprocessing layer
  - `stock_predictor.py`: Machine learning prediction engine
  - `chart_generator.py`: Visualization and charting component
- **Data Processing Pipeline**: Sequential data flow from fetching → feature engineering → prediction → visualization
- **Error Handling**: Comprehensive validation at each stage with user-friendly error messages

### Machine Learning Architecture
- **Primary Model**: Random Forest Regressor with optimized hyperparameters (100 estimators, max depth 10)
- **Feature Engineering**: Technical indicators including moving averages, price ratios, volatility measures, and momentum indicators
- **Data Preprocessing**: StandardScaler for feature normalization and handling of missing values
- **Validation**: Multiple accuracy metrics (MSE, MAE, R²) for model performance assessment

### Data Management
- **Real-time Data**: Yahoo Finance API integration via yfinance library for live market data
- **Historical Analysis**: Configurable time periods (default 365 days) for training data
- **Data Validation**: Comprehensive checks for data completeness and column requirements
- **Caching Strategy**: Streamlit's built-in caching for improved performance on repeated queries

### Visualization System
- **Charting Library**: Plotly for interactive, responsive charts with zoom and pan capabilities
- **Chart Types**: Candlestick charts with volume indicators and prediction overlays
- **Theme Consistency**: Dark theme implementation across all visual components
- **Responsive Design**: Adaptive layouts that work across different screen sizes

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework and UI components
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **plotly**: Interactive charting and data visualization
- **scikit-learn**: Machine learning algorithms and preprocessing tools
- **yfinance**: Yahoo Finance API wrapper for stock market data

### Data Sources
- **Yahoo Finance**: Primary data source for real-time and historical stock prices, market indicators, and company information
- **Global Market Support**: Access to major stock exchanges worldwide through Yahoo Finance's comprehensive database

### Machine Learning Stack
- **Random Forest**: Primary prediction algorithm for non-linear pattern recognition
- **Linear Models**: Alternative regression approaches for comparison and ensemble methods
- **Feature Engineering**: Technical analysis indicators and statistical transformations

### Development Tools
- **Warnings Filter**: Error suppression for cleaner user experience
- **DateTime Handling**: Time series data manipulation and date range calculations