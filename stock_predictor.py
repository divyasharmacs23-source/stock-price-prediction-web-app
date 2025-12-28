import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        """Initialize the StockPredictor with ML models."""
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.last_accuracy = {}
        
    def create_features(self, data):
        """
        Create technical indicator features for ML model.
        
        Args:
            data (pd.DataFrame): Stock price data with OHLCV columns
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        df = data.copy()
        
        # Price-based features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Moving average ratios
        df['SMA_5_20_ratio'] = df['SMA_5'] / df['SMA_20']
        df['SMA_10_20_ratio'] = df['SMA_10'] / df['SMA_20']
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
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
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Technical patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        # Time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        return df
    
    def prepare_training_data(self, data, target_days=1):
        """
        Prepare training data for the ML model.
        
        Args:
            data (pd.DataFrame): Stock data with features
            target_days (int): Number of days ahead to predict
            
        Returns:
            tuple: (X, y) training data
        """
        # Create target variable (future price)
        data['Target'] = data['Close'].shift(-target_days)
        
        # Select feature columns (exclude price columns and target)
        exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Remove rows with NaN values
        clean_data = data[feature_columns + ['Target']].dropna()
        
        X = clean_data[feature_columns]
        y = clean_data['Target']
        
        self.feature_columns = feature_columns
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train the ML model.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
        """
        if len(X) < 30:  # Need minimum data for training
            raise ValueError("Insufficient data for training. Need at least 30 data points.")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_scaled)
        self.last_accuracy = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
    
    def predict_prices(self, data, days_ahead=7):
        """
        Predict future stock prices.
        
        Args:
            data (pd.DataFrame): Historical stock data
            days_ahead (int): Number of days to predict
            
        Returns:
            np.array: Predicted prices
        """
        try:
            # Create features
            df_with_features = self.create_features(data)
            
            # Prepare training data
            X, y = self.prepare_training_data(df_with_features, target_days=1)
            
            if len(X) < 30:
                raise ValueError("Insufficient data for prediction")
            
            # Train the model
            self.train_model(X, y)
            
            predictions = []
            current_data = df_with_features.copy()
            
            for day in range(days_ahead):
                # Get the latest features
                latest_features = current_data[self.feature_columns].iloc[-1:].dropna(axis=1)
                
                # Ensure we have the same features as training
                common_features = [col for col in self.feature_columns if col in latest_features.columns]
                latest_features = latest_features[common_features]
                
                if latest_features.empty or len(common_features) < 10:
                    # If we don't have enough features, use simple trend prediction
                    recent_prices = data['Close'].tail(5)
                    trend = recent_prices.pct_change().mean()
                    last_price = current_data['Close'].iloc[-1]
                    predicted_price = last_price * (1 + trend)
                else:
                    # Scale features
                    latest_features_scaled = self.scaler.transform(latest_features)
                    
                    # Make prediction
                    predicted_price = self.model.predict(latest_features_scaled)[0]
                
                predictions.append(predicted_price)
                
                # Update current_data with the prediction for next iteration
                new_row = current_data.iloc[-1].copy()
                new_row['Close'] = predicted_price
                new_row['High'] = max(predicted_price, new_row['High'])
                new_row['Low'] = min(predicted_price, new_row['Low'])
                
                # Add the new row to current_data
                new_index = current_data.index[-1] + pd.Timedelta(days=1)
                current_data.loc[new_index] = new_row
                
                # Recalculate features with the new data
                current_data = self.create_features(current_data)
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
    
    def get_model_accuracy(self):
        """
        Get the last calculated model accuracy metrics.
        
        Returns:
            dict: Accuracy metrics
        """
        return self.last_accuracy if self.is_trained else {}
    
    def predict_trend(self, data, days=5):
        """
        Predict the general trend direction.
        
        Args:
            data (pd.DataFrame): Stock data
            days (int): Number of days to analyze
            
        Returns:
            dict: Trend prediction with confidence
        """
        try:
            predictions = self.predict_prices(data, days)
            if predictions is None:
                return None
            
            current_price = data['Close'].iloc[-1]
            avg_prediction = np.mean(predictions)
            
            trend_direction = "Bullish" if avg_prediction > current_price else "Bearish"
            price_change_pct = ((avg_prediction - current_price) / current_price) * 100
            
            # Calculate confidence based on prediction consistency
            prediction_std = np.std(predictions)
            confidence = max(0, min(100, 100 - (prediction_std / current_price * 100)))
            
            return {
                'direction': trend_direction,
                'confidence': confidence,
                'price_change_percent': price_change_pct,
                'predicted_price': avg_prediction,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"Trend prediction error: {str(e)}")
            return None
