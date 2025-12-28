import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataFetcher:
    def __init__(self):
        """Initialize the DataFetcher class."""
        pass
    
    def get_stock_data(self, symbol, days=365):
        """
        Fetch stock data for a given symbol and number of days with enhanced error handling.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'RELIANCE.NS')
            days (int): Number of days of historical data to fetch
            
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Validate symbol format
            if not symbol or len(symbol.strip()) == 0:
                st.error("Invalid symbol provided")
                return None
            
            symbol = symbol.strip().upper()
            
            # Calculate start date with buffer for weekends/holidays
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)  # Add buffer for weekends
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # First check if the ticker is valid
            try:
                ticker_info = ticker.info
                if not ticker_info or 'symbol' not in ticker_info:
                    st.error(f"Symbol {symbol} not found or may be delisted")
                    return None
            except:
                # If info fails, try to get minimal data first
                pass
            
            # Fetch historical data
            stock_data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True,
                prepost=True
            )
            
            if stock_data.empty:
                st.error(f"No trading data available for {symbol}. Symbol may be delisted or invalid.")
                return None
            
            # Get the most recent data (exactly 'days' worth if available)
            stock_data = stock_data.tail(days)
            
            # Clean the data
            stock_data = stock_data.dropna()
            
            if len(stock_data) < 10:  # Need minimum data points
                st.error(f"Insufficient data for {symbol}. Only {len(stock_data)} data points available.")
                return None
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in stock_data.columns for col in required_columns):
                st.error(f"Missing required price data columns for {symbol}")
                return None
            
            # Validate data quality
            if stock_data['Close'].isna().all():
                st.error(f"No valid price data for {symbol}")
                return None
            
            return stock_data
            
        except Exception as e:
            error_msg = str(e)
            if "possibly delisted" in error_msg.lower():
                st.error(f"⚠️ {symbol} appears to be delisted or suspended from trading")
            elif "no timezone" in error_msg.lower():
                st.error(f"⚠️ {symbol} has timezone issues - may be delisted or invalid")
            else:
                st.error(f"❌ Error fetching data for {symbol}: {error_msg}")
            return None
    
    def get_stock_info(self, symbol):
        """
        Get detailed information about a stock with enhanced error handling.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid info
            if not info or len(info) < 3:
                return {
                    'longName': symbol,
                    'sector': 'N/A',
                    'industry': 'N/A',
                    'marketCap': 'N/A',
                    'trailingPE': 'N/A',
                    'dividendYield': 'N/A',
                    'beta': 'N/A',
                    'currency': 'USD' if not symbol.endswith('.NS') else 'INR',
                    'exchange': 'NSE' if symbol.endswith('.NS') else 'NASDAQ',
                    'website': 'N/A',
                    'businessSummary': 'N/A'
                }
            
            # Return relevant information with fallbacks
            return {
                'longName': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap', 'N/A'),
                'trailingPE': info.get('trailingPE', 'N/A'),
                'dividendYield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A'),
                'currency': info.get('currency', 'USD' if not symbol.endswith('.NS') else 'INR'),
                'exchange': info.get('exchange', info.get('fullExchangeName', 'N/A')),
                'website': info.get('website', 'N/A'),
                'businessSummary': info.get('businessSummary', 'N/A')
            }
            
        except Exception as e:
            # Return basic info even if detailed fetch fails
            return {
                'longName': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'marketCap': 'N/A',
                'trailingPE': 'N/A',
                'dividendYield': 'N/A',
                'beta': 'N/A',
                'currency': 'USD' if not symbol.endswith('.NS') else 'INR',
                'exchange': 'NSE' if symbol.endswith('.NS') else 'Unknown',
                'website': 'N/A',
                'businessSummary': 'N/A'
            }
    
    def get_multiple_stocks_data(self, symbols, days=365):
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols (list): List of stock symbols
            days (int): Number of days of historical data
            
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        stocks_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, days)
            if data is not None:
                stocks_data[symbol] = data
        
        return stocks_data
    
    def get_real_time_price(self, symbol):
        """
        Get real-time price for a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Real-time price information
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                latest = data.iloc[-1]
                return {
                    'price': latest['Close'],
                    'volume': latest['Volume'],
                    'timestamp': data.index[-1]
                }
            else:
                return None
                
        except Exception as e:
            st.warning(f"Could not fetch real-time data for {symbol}: {str(e)}")
            return None
    
    def validate_symbol(self, symbol):
        """
        Validate if a stock symbol exists and is tradeable.
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to fetch 1 day of data
            data = ticker.history(period='1d')
            return not data.empty
        except:
            return False
    
    def get_market_indices(self):
        """
        Get data for major market indices.
        
        Returns:
            dict: Market indices data
        """
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'Nifty 50': '^NSEI',
            'Sensex': '^BSESN',
            'Nikkei': '^N225',
            'FTSE 100': '^FTSE'
        }
        
        indices_data = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d')
                if not data.empty:
                    latest = data.iloc[-1]
                    indices_data[name] = {
                        'symbol': symbol,
                        'price': latest['Close'],
                        'change': latest['Close'] - latest['Open'],
                        'change_percent': ((latest['Close'] - latest['Open']) / latest['Open']) * 100
                    }
            except:
                continue
        
        return indices_data
    
    def search_stocks(self, query, limit=10):
        """
        Search for stocks based on a query.
        Note: This is a basic implementation. In a production environment,
        you might want to use a more sophisticated search API.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            list: List of matching stock symbols
        """
        # Common Indian stock symbols
        indian_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HDFC.NS',
            'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'HINDUNILVR.NS',
            'LT.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'BAJFINANCE.NS', 'AXISBANK.NS',
            'KOTAKBANK.NS', 'TITAN.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'ULTRACEMCO.NS'
        ]
        
        # Common global stock symbols
        global_stocks = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'NFLX',
            'BRK-B', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'ADBE',
            'CRM', 'CMCSA', 'XOM', 'BAC', 'VZ', 'ABT', 'PFE', 'T', 'WMT', 'MRK'
        ]
        
        all_stocks = indian_stocks + global_stocks
        
        # Simple search based on symbol matching
        matching_stocks = []
        query_upper = query.upper()
        
        for stock in all_stocks:
            if query_upper in stock.upper():
                matching_stocks.append(stock)
                if len(matching_stocks) >= limit:
                    break
        
        return matching_stocks
