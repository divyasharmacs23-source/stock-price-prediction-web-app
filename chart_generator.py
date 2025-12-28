import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ChartGenerator:
    def __init__(self):
        """Initialize the ChartGenerator class."""
        self.dark_theme = {
            'bgcolor': '#0e1117',
            'paper_bgcolor': '#0e1117',
            'font_color': '#fafafa',
            'gridcolor': '#262730',
            'line_color': '#fafafa'
        }
    
    def apply_dark_theme(self, fig):
        """Apply dark theme to a plotly figure."""
        fig.update_layout(
            plot_bgcolor=self.dark_theme['bgcolor'],
            paper_bgcolor=self.dark_theme['paper_bgcolor'],
            font_color=self.dark_theme['font_color'],
            xaxis=dict(gridcolor=self.dark_theme['gridcolor']),
            yaxis=dict(gridcolor=self.dark_theme['gridcolor'])
        )
        return fig
    
    def create_candlestick_chart(self, data, title):
        """
        Create an interactive candlestick chart.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Candlestick chart
        """
        # Create subplots with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=[f'{title} - Price', 'Volume'],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing=dict(line=dict(color='#00ff00')),
                decreasing=dict(line=dict(color='#ff4444'))
            ),
            row=1, col=1
        )
        
        # Add moving averages
        data_with_ma = self.add_technical_indicators(data)
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_ma['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#00d4aa', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_ma['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#ff6b6b', width=1)
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['#00ff00' if close >= open else '#ff4444' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{title} - Interactive Candlestick Chart',
                x=0.5,
                font=dict(size=20, color='#fafafa')
            ),
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Apply dark theme
        fig = self.apply_dark_theme(fig)
        
        return fig
    
    def create_line_chart(self, data, title):
        """
        Create a line chart for stock prices.
        
        Args:
            data (pd.DataFrame): Stock data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Line chart
        """
        fig = go.Figure()
        
        # Close price line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00d4aa', width=2),
                hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
            )
        )
        
        # Add technical indicators
        data_with_indicators = self.add_technical_indicators(data)
        
        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#ff6b6b', width=1, dash='dash')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#4ecdc4', width=1, dash='dash')
            )
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='#ffa500', width=1),
                opacity=0.5
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='#ffa500', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.1)',
                opacity=0.5
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{title} - Price Movement with Technical Indicators',
                x=0.5,
                font=dict(size=20, color='#fafafa')
            ),
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True
        )
        
        # Apply dark theme
        fig = self.apply_dark_theme(fig)
        
        return fig
    
    def create_ohlc_chart(self, data, title):
        """
        Create an OHLC (Open, High, Low, Close) chart.
        
        Args:
            data (pd.DataFrame): Stock data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: OHLC chart
        """
        fig = go.Figure(data=go.Ohlc(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing=dict(line=dict(color='#00ff00')),
            decreasing=dict(line=dict(color='#ff4444'))
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{title} - OHLC Chart',
                x=0.5,
                font=dict(size=20, color='#fafafa')
            ),
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600
        )
        
        # Apply dark theme
        fig = self.apply_dark_theme(fig)
        
        return fig
    
    def create_volume_chart(self, data, title):
        """
        Create a volume analysis chart.
        
        Args:
            data (pd.DataFrame): Stock data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Volume chart
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.05,
            subplot_titles=[f'{title} - Price', 'Volume Analysis']
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#00d4aa', width=2)
            ),
            row=1, col=1
        )
        
        # Volume bars with colors based on price movement
        colors = ['#00ff00' if close >= open else '#ff4444' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.8
            ),
            row=2, col=1
        )
        
        # Volume moving average
        volume_ma = data['Volume'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volume_ma,
                mode='lines',
                name='Volume MA 20',
                line=dict(color='#ffa500', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{title} - Price and Volume Analysis',
                x=0.5,
                font=dict(size=20, color='#fafafa')
            ),
            height=700,
            showlegend=True
        )
        
        # Apply dark theme
        fig = self.apply_dark_theme(fig)
        
        return fig
    
    def create_prediction_chart(self, historical_data, predictions, days_ahead, title):
        """
        Create a chart showing historical data and predictions.
        
        Args:
            historical_data (pd.DataFrame): Historical stock data
            predictions (np.array): Predicted prices
            days_ahead (int): Number of days predicted
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Prediction chart
        """
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='#00d4aa', width=2)
            )
        )
        
        # Create future dates
        last_date = historical_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Predictions',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                marker=dict(size=6, color='#ff6b6b')
            )
        )
        
        # Connection line between last historical point and first prediction
        fig.add_trace(
            go.Scatter(
                x=[historical_data.index[-1], future_dates[0]],
                y=[historical_data['Close'].iloc[-1], predictions[0]],
                mode='lines',
                name='Transition',
                line=dict(color='#ffa500', width=2, dash='dot'),
                showlegend=False
            )
        )
        
        # Add confidence bands (simple implementation)
        std_dev = np.std(historical_data['Close'].pct_change().dropna()) * historical_data['Close'].iloc[-1]
        upper_bound = predictions + std_dev
        lower_bound = predictions - std_dev
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper_bound,
                mode='lines',
                name='Upper Confidence',
                line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower_bound,
                mode='lines',
                name='Lower Confidence',
                line=dict(color='rgba(255, 107, 107, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 107, 107, 0.2)',
                showlegend=True
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{title} - Price Prediction ({days_ahead} Days)',
                x=0.5,
                font=dict(size=20, color='#fafafa')
            ),
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True
        )
        
        # Apply dark theme
        fig = self.apply_dark_theme(fig)
        
        return fig
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to stock data.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def create_technical_indicators_chart(self, data, title):
        """
        Create a comprehensive technical indicators chart.
        
        Args:
            data (pd.DataFrame): Stock data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Technical indicators chart
        """
        data_with_indicators = self.add_technical_indicators(data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.05,
            subplot_titles=[f'{title} - Price & Moving Averages', 'MACD', 'RSI']
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close',
                line=dict(color='#00d4aa', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#ff6b6b', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#4ecdc4', width=1)
            ),
            row=1, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#00d4aa', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='#ff6b6b', width=1)
            ),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data_with_indicators['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#ffa500', width=2)
            ),
            row=3, col=1
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{title} - Technical Analysis',
                x=0.5,
                font=dict(size=20, color='#fafafa')
            ),
            height=800,
            showlegend=True
        )
        
        # Apply dark theme
        fig = self.apply_dark_theme(fig)
        
        return fig
