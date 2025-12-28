import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from stock_predictor import StockPredictor
from data_fetcher import DataFetcher
from chart_generator import ChartGenerator

# Configure page
st.set_page_config(
    page_title="Global Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling for enhanced visibility and appearance
st.markdown("""
<style>
    /* Main metric cards */
    .stMetric > div > div > div > div {
        background-color: #262730;
        border: 1px solid #00d4aa;
        padding: 15px;
        border-radius: 12px;
        color: #fafafa !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stMetric > div > div > div > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 212, 170, 0.2);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div > select {
        background-color: #262730;
        color: #fafafa;
        border: 2px solid #00d4aa;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Prediction boxes */
    .prediction-box {
        background: linear-gradient(135deg, #262730 0%, #1a1a1a 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #00d4aa;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    
    /* Stock information cards */
    .stock-info {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #00d4aa;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: #fafafa;
    }
    
    .stock-info h4, .stock-info h5 {
        color: #00d4aa;
        margin-bottom: 10px;
        font-weight: 600;
    }
    
    /* High/Low price containers */
    .high-low-container {
        background: linear-gradient(135deg, #262730 0%, #1a1a1a 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #444;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    
    /* Professional headers */
    .main-header {
        background: linear-gradient(90deg, #00d4aa 0%, #0099cc 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    /* Feature highlights */
    .feature-highlight {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00d4aa;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Remove any white backgrounds */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Ensure all text is visible */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #00d4aa 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher()
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StockPredictor()
    if 'chart_generator' not in st.session_state:
        st.session_state.chart_generator = ChartGenerator()

    # Professional header
    st.title("📈 Professional Stock Market Analysis & Prediction Platform")
    st.markdown("### Real-time Global Stock Data with AI-Powered Price Predictions")
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🌍 Global Coverage**<br/>Worldwide stock markets", unsafe_allow_html=True)
    with col2:
        st.markdown("**🇮🇳 Indian Focus**<br/>60+ Indian stocks", unsafe_allow_html=True)
    with col3:
        st.markdown("**📊 80-Day Analysis**<br/>Comprehensive data", unsafe_allow_html=True)
    with col4:
        st.markdown("**🤖 AI Predictions**<br/>Machine learning powered", unsafe_allow_html=True)
    
    st.markdown("---")

    # Sidebar for stock selection
    with st.sidebar:
        st.header("📊 Stock Selection")
        
        # Comprehensive Indian stocks list (verified symbols)
        indian_stocks = {
            "Reliance Industries": "RELIANCE.NS",
            "Tata Consultancy Services": "TCS.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "State Bank of India": "SBIN.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "ITC": "ITC.NS",
            "Hindustan Unilever": "HINDUNILVR.NS",
            "Larsen & Toubro": "LT.NS",
            "Asian Paints": "ASIANPAINT.NS",
            "Maruti Suzuki": "MARUTI.NS",
            "Bajaj Finance": "BAJFINANCE.NS",
            "Axis Bank": "AXISBANK.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Titan Company": "TITAN.NS",
            "Nestle India": "NESTLEIND.NS",
            "Wipro": "WIPRO.NS",
            "UltraTech Cement": "ULTRACEMCO.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS",
            "Sun Pharmaceutical": "SUNPHARMA.NS",
            "Tech Mahindra": "TECHM.NS",
            "ONGC": "ONGC.NS",
            "PowerGrid Corp": "POWERGRID.NS",
            "NTPC": "NTPC.NS",
            "Coal India": "COALINDIA.NS",
            "Bajaj Finserv": "BAJAJFINSV.NS",
            "Tata Steel": "TATASTEEL.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS",
            "IndusInd Bank": "INDUSINDBK.NS",
            "Dr Reddy's Labs": "DRREDDY.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "Mahindra & Mahindra": "M&M.NS",
            "Grasim Industries": "GRASIM.NS",
            "Shree Cement": "SHREECEM.NS",
            "Cipla": "CIPLA.NS",
            "JSW Steel": "JSWSTEEL.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Adani Enterprises": "ADANIENT.NS",
            "Adani Ports": "ADANIPORTS.NS",
            "BPCL": "BPCL.NS",
            "IOCL": "IOC.NS",
            "Hindalco": "HINDALCO.NS",
            "Britannia Industries": "BRITANNIA.NS",
            "Divis Labs": "DIVISLAB.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Vedanta": "VEDL.NS",
            "Apollo Hospitals": "APOLLOHOSP.NS",
            "Godrej Consumer": "GODREJCP.NS",
            "Pidilite Industries": "PIDILITIND.NS",
            "Dabur India": "DABUR.NS",
            "Colgate Palmolive": "COLPAL.NS",
            "Marico": "MARICO.NS",
            "Biocon": "BIOCON.NS",
            "Aurobindo Pharma": "AUROPHARMA.NS",
            "Lupin": "LUPIN.NS",
            "Cadila Healthcare": "CADILAHC.NS"
        }
        
        # Comprehensive Global stocks list
        global_stocks = {
            # US Tech Giants
            "Apple Inc": "AAPL",
            "Microsoft": "MSFT",
            "Amazon": "AMZN",
            "Google (Alphabet)": "GOOGL",
            "Tesla": "TSLA",
            "Meta (Facebook)": "META",
            "NVIDIA": "NVDA",
            "Netflix": "NFLX",
            "Adobe": "ADBE",
            "Salesforce": "CRM",
            "PayPal": "PYPL",
            "Intel": "INTC",
            "IBM": "IBM",
            "Oracle": "ORCL",
            "Cisco": "CSCO",
            
            # US Financial & Traditional
            "Berkshire Hathaway": "BRK-B",
            "JPMorgan Chase": "JPM",
            "Bank of America": "BAC",
            "Visa": "V",
            "Mastercard": "MA",
            "Johnson & Johnson": "JNJ",
            "Procter & Gamble": "PG",
            "Walmart": "WMT",
            "Coca-Cola": "KO",
            "McDonald's": "MCD",
            "Disney": "DIS",
            "Nike": "NKE",
            
            # International
            "Samsung Electronics": "005930.KS",
            "Toyota Motor": "TM",
            "ASML Holding": "ASML",
            "Taiwan Semiconductor": "TSM",
            "Alibaba Group": "BABA",
            "Tencent Holdings": "0700.HK",
            "LVMH": "MC.PA",
            "Nestle": "NESN.SW",
            "TSMC": "2330.TW",
            "Roche Holdings": "ROG.SW"
        }
        
        # Stock selection options
        selection_type = st.radio(
            "Select Stock Category:",
            ["Indian Stocks (NSE)", "Global Stocks", "Custom Symbol"]
        )
        
        if selection_type == "Indian Stocks (NSE)":
            stock_name = st.selectbox("Choose Indian Stock:", list(indian_stocks.keys()))
            symbol = indian_stocks[stock_name]
        elif selection_type == "Global Stocks":
            stock_name = st.selectbox("Choose Global Stock:", list(global_stocks.keys()))
            symbol = global_stocks[stock_name]
        else:
            symbol = st.text_input("Enter Stock Symbol:", placeholder="e.g., AAPL, RELIANCE.NS")
            stock_name = symbol
        
        # Fixed 80 days period as requested
        days = 80
        st.info("📅 Using 80 days of historical data for analysis and prediction")
        
        # Professional chart options
        st.subheader("📊 Analysis Options")
        chart_type = st.selectbox(
            "Chart Type:",
            ["Candlestick", "Line Chart", "OHLC", "Volume Analysis", "Technical Indicators"]
        )
        
        # Analysis timeframe
        st.info("📅 **Analysis Period:** 80 trading days for optimal accuracy")
        
        # Prediction settings
        st.subheader("🔮 Prediction Settings")
        predict_days = st.slider("Days to Predict:", 1, 30, 7)
        
        # Fetch data button
        fetch_button = st.button("📈 Analyze Stock", type="primary", use_container_width=True)

    # Main content area
    if fetch_button and symbol:
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                # Fetch stock data
                stock_data = st.session_state.data_fetcher.get_stock_data(symbol, days)
                
                if stock_data is None or stock_data.empty:
                    st.error(f"❌ Unable to fetch data for symbol: {symbol}")
                    st.info("Please check if the symbol is correct and try again.")
                    return
                
                # Get stock info
                stock_info = st.session_state.data_fetcher.get_stock_info(symbol)
                
                # Display stock information
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"₹{current_price:.2f}" if symbol.endswith('.NS') else f"${current_price:.2f}",
                        delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
                    )
                
                with col2:
                    market_cap = stock_info.get('marketCap', 'N/A')
                    if market_cap != 'N/A':
                        market_cap = f"{market_cap/1e9:.2f}B" if market_cap > 1e9 else f"{market_cap/1e6:.2f}M"
                    st.metric(label="Market Cap", value=market_cap)
                
                with col3:
                    volume = stock_data['Volume'].iloc[-1]
                    avg_volume = stock_data['Volume'].mean()
                    volume_change = ((volume - avg_volume) / avg_volume) * 100
                    st.metric(
                        label="Volume",
                        value=f"{volume:,.0f}",
                        delta=f"{volume_change:+.1f}% vs avg"
                    )
                
                with col4:
                    pe_ratio = stock_info.get('trailingPE', 'N/A')
                    if pe_ratio != 'N/A':
                        pe_ratio = f"{pe_ratio:.2f}"
                    st.metric(label="P/E Ratio", value=pe_ratio)
                
                # Professional stock information display
                if stock_info:
                    st.markdown("### 🏢 Company Overview")
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.markdown(f"""
                        <div class="stock-info">
                            <h4>🏢 {stock_info.get('longName', stock_name)}</h4>
                            <p><strong>🏦 Sector:</strong> {stock_info.get('sector', 'N/A')}</p>
                            <p><strong>🏭 Industry:</strong> {stock_info.get('industry', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with info_col2:
                        st.markdown(f"""
                        <div class="stock-info">
                            <h4>💼 Trading Information</h4>
                            <p><strong>🏪 Exchange:</strong> {stock_info.get('exchange', 'N/A')}</p>
                            <p><strong>💰 Currency:</strong> {stock_info.get('currency', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Calculate and display highest and lowest prices with dates
                highest_price = stock_data['High'].max()
                lowest_price = stock_data['Low'].min()
                highest_date = stock_data.loc[stock_data['High'].idxmax()].name
                lowest_date = stock_data.loc[stock_data['Low'].idxmin()].name
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="high-low-container">
                        <h4 style="color: #00ff00;">📈 Highest Price in Period</h4>
                        <h2 style="color: #00ff00;">{'₹' if symbol.endswith('.NS') else '$'}{highest_price:.2f}</h2>
                        <p><strong>Date:</strong> {highest_date.strftime('%B %d, %Y')}</p>
                        <p><strong>Time:</strong> {highest_date.strftime('%I:%M %p')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="high-low-container">
                        <h4 style="color: #ff4444;">📉 Lowest Price in Period</h4>
                        <h2 style="color: #ff4444;">{'₹' if symbol.endswith('.NS') else '$'}{lowest_price:.2f}</h2>
                        <p><strong>Date:</strong> {lowest_date.strftime('%B %d, %Y')}</p>
                        <p><strong>Time:</strong> {lowest_date.strftime('%I:%M %p')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate charts
                st.subheader(f"📊 {chart_type} - {stock_name}")
                
                if chart_type == "Candlestick":
                    fig = st.session_state.chart_generator.create_candlestick_chart(stock_data, stock_name)
                elif chart_type == "Line Chart":
                    fig = st.session_state.chart_generator.create_line_chart(stock_data, stock_name)
                elif chart_type == "OHLC":
                    fig = st.session_state.chart_generator.create_ohlc_chart(stock_data, stock_name)
                elif chart_type == "Volume Analysis":
                    fig = st.session_state.chart_generator.create_volume_chart(stock_data, stock_name)
                else:  # Technical Indicators
                    fig = st.session_state.chart_generator.create_technical_indicators_chart(stock_data, stock_name)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Analysis
                st.subheader("📊 Technical Analysis")
                
                # Calculate technical indicators
                stock_data_with_indicators = st.session_state.chart_generator.add_technical_indicators(stock_data)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sma_20 = stock_data_with_indicators['SMA_20'].iloc[-1]
                    st.metric(
                        label="SMA (20)",
                        value=f"{'₹' if symbol.endswith('.NS') else '$'}{sma_20:.2f}",
                        delta=f"{((current_price - sma_20) / sma_20 * 100):+.2f}%"
                    )
                
                with col2:
                    sma_50 = stock_data_with_indicators['SMA_50'].iloc[-1]
                    st.metric(
                        label="SMA (50)",
                        value=f"{'₹' if symbol.endswith('.NS') else '$'}{sma_50:.2f}",
                        delta=f"{((current_price - sma_50) / sma_50 * 100):+.2f}%"
                    )
                
                with col3:
                    rsi = stock_data_with_indicators['RSI'].iloc[-1]
                    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric(
                        label="RSI (14)",
                        value=f"{rsi:.2f}",
                        delta=rsi_signal
                    )
                
                with col4:
                    volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric(
                        label="Volatility (Annual)",
                        value=f"{volatility:.2f}%"
                    )
                
                # Stock Price Prediction
                st.subheader("🔮 AI-Powered Price Prediction")
                
                with st.spinner("Training ML model and generating predictions..."):
                    try:
                        # Use exactly 80 days for prediction as requested
                        prediction_data = stock_data.tail(80)
                        predictions = st.session_state.predictor.predict_prices(
                            prediction_data, predict_days
                        )
                        
                        if predictions is not None:
                            # Display prediction results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                next_day_prediction = predictions[0]
                                prediction_change = next_day_prediction - current_price
                                prediction_change_pct = (prediction_change / current_price) * 100
                                
                                st.markdown(f"""   <div class="prediction-box">
                                    <h4>🎯 Next Day Prediction</h4>
                                    <h2 style="color: {'#00ff00' if prediction_change > 0 else '#ff4444'};">
                                        {'₹' if symbol.endswith('.NS') else '$'}{next_day_prediction:.2f}
                                    </h2>
                                    <p><strong>Expected Change:</strong> 
                                    <span style="color: {'#00ff00' if prediction_change > 0 else '#ff4444'};">
                                    {prediction_change:+.2f} ({prediction_change_pct:+.2f}%)
                                    </span></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                avg_prediction = np.mean(predictions)
                                avg_change = avg_prediction - current_price
                                avg_change_pct = (avg_change / current_price) * 100
                                
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h4>📊 {predict_days}-Day Average Prediction</h4>
                                    <h2 style="color: {'#00ff00' if avg_change > 0 else '#ff4444'};">
                                        {'₹' if symbol.endswith('.NS') else '$'}{avg_prediction:.2f}
                                    </h2>
                                    <p><strong>Expected Change:</strong> 
                                    <span style="color: {'#00ff00' if avg_change > 0 else '#ff4444'};">
                                    {avg_change:+.2f} ({avg_change_pct:+.2f}%)
                                    </span></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Create prediction chart
                            prediction_fig = st.session_state.chart_generator.create_prediction_chart(
                                stock_data, predictions, predict_days, stock_name
                            )
                            st.plotly_chart(prediction_fig, use_container_width=True)
                            
                            # Model performance metrics
                            accuracy_metrics = st.session_state.predictor.get_model_accuracy()
                            if accuracy_metrics:
                                st.subheader("🎯 Model Performance")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("R² Score", f"{accuracy_metrics.get('r2_score', 0):.4f}")
                                with col2:
                                    st.metric("RMSE", f"{accuracy_metrics.get('rmse', 0):.4f}")
                                with col3:
                                    st.metric("MAE", f"{accuracy_metrics.get('mae', 0):.4f}")
                        
                        else:
                            st.error("❌ Unable to generate predictions. Insufficient data.")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                
                # Professional market insights
                st.subheader("📊 Professional Market Analysis")
                
                # Enhanced analysis
                recent_trend = "Bullish" if price_change > 0 else "Bearish"
                trend_strength = "Strong" if abs(price_change_pct) > 2 else "Moderate" if abs(price_change_pct) > 1 else "Weak"
                
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.markdown(f"""
                    <div class="stock-info">
                        <h5>📈 Price Movement Analysis</h5>
                        <p><strong>Trend:</strong> {trend_strength} {recent_trend} ({price_change_pct:+.2f}%)</p>
                        <p><strong>SMA Position:</strong> {'Above' if current_price > sma_20 else 'Below'} 20-day average</p>
                        <p><strong>Market Sentiment:</strong> {rsi_signal} (RSI: {rsi:.1f})</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with insight_col2:
                    st.markdown(f"""
                    <div class="stock-info">
                        <h5>📊 Risk Assessment</h5>
                        <p><strong>Volatility:</strong> {'High' if volatility > 30 else 'Moderate' if volatility > 15 else 'Low'} ({volatility:.1f}%)</p>
                        <p><strong>Volume Activity:</strong> {volume_change:+.1f}% vs average</p>
                        <p><strong>Analysis Period:</strong> 80 trading days</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.info("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
