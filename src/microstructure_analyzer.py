import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MicrostructureAnalyzer:
    """Advanced market microstructure analysis"""
    
    def __init__(self):
        self.gmm_model = GaussianMixture(n_components=3, covariance_type='full')
        self.regime_history = []
        
    def detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """Detect market regime using Gaussian Mixture Model"""
        
        # Feature engineering for regime detection
        features = pd.DataFrame()
        
        # Volatility clustering
        features['volatility'] = data['returns'].rolling(20).std()
        features['volatility_change'] = features['volatility'].pct_change()
        
        # Volume patterns
        features['volume_intensity'] = data['Volume'] / data['Volume'].rolling(50).mean()
        
        # Price efficiency
        features['price_efficiency'] = self.calculate_price_efficiency(data)
        
        # Microstructure noise
        features['noise_ratio'] = self.estimate_microstructure_noise(data)
        
        # Clean data
        features = features.dropna()
        
        # Fit GMM
        self.gmm_model.fit(features)
        
        # Predict regime
        current_regime = self.gmm_model.predict(features.iloc[-1:].values)[0]
        regime_probs = self.gmm_model.predict_proba(features.iloc[-1:].values)[0]
        
        # Regime characteristics
        regimes = {
            0: "Low Volatility",
            1: "High Volatility", 
            2: "Transition"
        }
        
        return {
            'current_regime': regimes[current_regime],
            'regime_probabilities': dict(zip(regimes.values(), regime_probs)),
            'regime_stability': self.calculate_regime_stability(features),
            'regime_duration': self.estimate_regime_duration(current_regime)
        }
    
    def calculate_price_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Hurst exponent for price efficiency"""
        
        window = 100
        efficiency = pd.Series(index=data.index)
        
        for i in range(window, len(data)):
            prices = data['Close'].iloc[i-window:i].values
            
            # R/S analysis
            mean_price = np.mean(prices)
            deviations = prices - mean_price
            cumsum = np.cumsum(deviations)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(prices)
            
            if S != 0:
                RS = R / S
                efficiency.iloc[i] = np.log(RS) / np.log(window)
            else:
                efficiency.iloc[i] = 0.5
        
        return efficiency
    
    def estimate_microstructure_noise(self, data: pd.DataFrame) -> pd.Series:
        """Estimate microstructure noise using Zhang et al. (2005) method"""
        
        # Two-timescale estimator
        fast_var = data['returns'].rolling(5).var()
        slow_var = data['returns'].rolling(20).var()
        
        # Noise ratio
        noise_ratio = np.maximum(0, (fast_var - slow_var) / slow_var)
        
        return noise_ratio.fillna(0)
    
    def detect_liquidity_shocks(self, data: pd.DataFrame, order_imbalance: float) -> Dict:
        """Detect liquidity provision shocks"""
        
        # Amihud illiquidity ratio
        illiquidity = np.abs(data['returns']) / (data['Volume'] * data['Close'])
        illiquidity_ma = illiquidity.rolling(20).mean()
        illiquidity_std = illiquidity.rolling(20).std()
        
        # Current illiquidity
        current_illiquidity = illiquidity.iloc[-1]
        
        # Z-score
        z_score = (current_illiquidity - illiquidity_ma.iloc[-1]) / illiquidity_std.iloc[-1]
        
        # Liquidity shock detection
        is_shock = np.abs(z_score) > 2
        
        return {
            'illiquidity_ratio': current_illiquidity,
            'liquidity_z_score': z_score,
            'is_liquidity_shock': is_shock,
            'shock_direction': 'dry_up' if z_score > 0 else 'excess',
            'order_book_imbalance': order_imbalance
        }
    
    def calculate_adverse_selection(self, trades: List[Dict], quotes: List[Dict]) -> float:
        """Calculate probability of informed trading"""
        
        if not trades or not quotes:
            return 0.0
        
        # Effective spread
        effective_spreads = []
        
        for trade in trades[-100:]:  # Last 100 trades
            # Find nearest quote
            trade_time = trade['timestamp']
            nearest_quote = min(quotes, key=lambda q: abs(q['timestamp'] - trade_time))
            
            # Midpoint
            midpoint = (nearest_quote['bid'] + nearest_quote['ask']) / 2
            
            # Effective spread
            eff_spread = 2 * abs(trade['price'] - midpoint) / midpoint
            effective_spreads.append(eff_spread)
        
        if not effective_spreads:
            return 0.0
        
        # Higher effective spread indicates adverse selection
        avg_eff_spread = np.mean(effective_spreads)
        
        # Normalize to probability
        prob_informed = min(1.0, avg_eff_spread * 100)  # Scale factor
        
        return prob_informed
    
    def calculate_kyle_lambda(self, data: pd.DataFrame) -> float:
        """Calculate Kyle's Lambda (price impact coefficient)"""
        
        # Need order flow data
        signed_volume = data['Volume'] * np.sign(data['returns'])
        
        # Regression: price change on signed volume
        X = signed_volume.values[-100:].reshape(-1, 1)
        y = data['returns'].values[-100:]
        
        # Remove NaN
        mask = ~(np.isnan(X.flatten()) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            return 0.0
        
        # Simple regression
        lambda_coef = np.linalg.lstsq(X, y, rcond=None)[0][0]
        
        return lambda_coef
    
    def calculate_regime_stability(self, features: pd.DataFrame) -> float:
        """Calculate regime stability score"""
        
        # Predict regimes for recent history
        recent_regimes = self.gmm_model.predict(features.tail(50).values)
        
        # Count regime changes
        regime_changes = np.sum(np.diff(recent_regimes) != 0)
        
        # Stability score (inverse of change frequency)
        stability = 1 - (regime_changes / len(recent_regimes))
        
        return stability
    
    def estimate_regime_duration(self, current_regime: int) -> int:
        """Estimate expected duration of current regime"""
        
        # This is simplified - in production, use survival analysis
        avg_durations = {0: 150, 1: 50, 2: 30}  # bars
        
        return avg_durations.get(current_regime, 100)