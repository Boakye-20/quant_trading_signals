"""
Signal Generator - Public Version
Demonstrates signal generation capabilities without revealing proprietary logic
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SignalGenerator:
    """
    Generates trading signals using technical and microstructure analysis
    
    Production Features (Not Included):
    - Neural network ensemble models
    - Advanced market microstructure analysis
    - Order flow toxicity detection
    - Multi-timeframe signal aggregation
    - Proprietary risk management algorithms
    """
    
    def __init__(self):
        self.confidence_threshold = 0.5
        self.signal_history = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize strategy parameters (simplified for demo)"""
        self.lookback_periods = {'short': 20, 'medium': 50, 'long': 200}
        self.volatility_window = 20
        self.rsi_period = 14
        
    def generate_signal(self, market_data: pd.DataFrame, 
                       microstructure_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate trading signal based on market data
        
        Parameters:
        -----------
        market_data: DataFrame with OHLCV data
        microstructure_data: Dict with market microstructure metrics (optional)
        
        Returns:
        --------
        Dict with signal details or None if no signal
        """
        
        if len(market_data) < self.lookback_periods['long']:
            return None
        
        # Calculate basic indicators
        indicators = self._calculate_indicators(market_data)
        
        # Generate base signal
        signal = self._generate_base_signal(indicators, market_data)
        
        if signal is None:
            return None
            
        # Apply filters (simplified)
        signal = self._apply_risk_filters(signal, market_data)
        
        # Add timestamp
        signal['timestamp'] = datetime.now()
        signal['symbol'] = market_data.index.name if market_data.index.name else 'UNKNOWN'
        
        # Store in history
        self.signal_history.append(signal)
        
        return signal
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators (simplified set)"""
        current_price = data['Close'].iloc[-1]
        
        # Moving averages
        sma_short = data['Close'].rolling(self.lookback_periods['short']).mean().iloc[-1]
        sma_medium = data['Close'].rolling(self.lookback_periods['medium']).mean().iloc[-1]
        sma_long = data['Close'].rolling(self.lookback_periods['long']).mean().iloc[-1]
        
        # RSI (simplified)
        rsi = self._calculate_rsi(data['Close'])
        
        # Volatility
        returns = data['Close'].pct_change()
        volatility = returns.rolling(self.volatility_window).std().iloc[-1] * np.sqrt(252)
        
        # Volume analysis
        volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
        
        return {
            'current_price': current_price,
            'sma_short': sma_short,
            'sma_medium': sma_medium,
            'sma_long': sma_long,
            'rsi': rsi,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'trend': 'UP' if sma_short > sma_medium > sma_long else 'DOWN'
        }
    
    def _generate_base_signal(self, indicators: Dict, data: pd.DataFrame) -> Optional[Dict]:
        """Generate base signal from indicators"""
        
        signal_strength = 0
        signal_type = 'HOLD'
        reasons = []
        
        # Trend following component
        if indicators['trend'] == 'UP' and indicators['rsi'] < 70:
            signal_strength += 0.3
            reasons.append("Uptrend confirmed")
        elif indicators['trend'] == 'DOWN' and indicators['rsi'] > 30:
            signal_strength -= 0.3
            reasons.append("Downtrend confirmed")
        
        # Mean reversion component
        price_deviation = (indicators['current_price'] - indicators['sma_short']) / indicators['sma_short']
        if abs(price_deviation) > 0.02:  # 2% deviation
            if price_deviation > 0 and indicators['rsi'] > 70:
                signal_strength -= 0.2
                reasons.append("Overbought condition")
            elif price_deviation < 0 and indicators['rsi'] < 30:
                signal_strength += 0.2
                reasons.append("Oversold condition")
        
        # Volume confirmation
        if indicators['volume_ratio'] > 1.5:
            signal_strength *= 1.2
            reasons.append("High volume confirmation")
        
        # Determine signal
        if signal_strength > 0.3:
            signal_type = 'BUY'
        elif signal_strength < -0.3:
            signal_type = 'SELL'
        else:
            return None
        
        # Calculate confidence (simplified)
        confidence = min(abs(signal_strength), 1.0)
        
        return {
            'action': signal_type,
            'confidence': confidence,
            'score': confidence * 100,
            'strength': confidence,
            'reasons': reasons,
            'indicators': indicators
        }
    
    def _apply_risk_filters(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Apply risk management filters"""
        
        volatility = signal['indicators']['volatility']
        current_price = signal['indicators']['current_price']
        
        # Position sizing (simplified Kelly Criterion)
        base_size = 0.02  # 2% base position
        volatility_adjustment = min(0.02 / volatility, 2.0)  # Target 2% volatility
        confidence_adjustment = signal['confidence']
        
        position_size = base_size * volatility_adjustment * confidence_adjustment
        position_size = min(position_size, 0.10)  # Max 10% position
        
        # Stop loss and take profit
        atr_multiplier = 2.0 if signal['action'] == 'BUY' else -2.0
        stop_distance = volatility * atr_multiplier
        
        if signal['action'] == 'BUY':
            stop_loss = current_price * (1 - abs(stop_distance))
            take_profit = current_price * (1 + abs(stop_distance) * 1.5)
        else:
            stop_loss = current_price * (1 + abs(stop_distance))
            take_profit = current_price * (1 - abs(stop_distance) * 1.5)
        
        signal.update({
            'position_size': round(position_size, 4),
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'current_price': round(current_price, 2),
            'risk_reward_ratio': 1.5,
            'max_risk_per_trade': 0.02  # 2% max risk
        })
        
        return signal
    
    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_period
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def generate_composite_signal(self, 
                                technical_signal: Dict, 
                                microstructure_signal: Dict, 
                                order_flow_signal: Dict) -> Dict:
        """
        Combine multiple signal types
        
        Note: Production system uses proprietary signal fusion algorithms
        including machine learning ensemble methods
        """
        
        # Weight components (simplified)
        weights = {
            'technical': 0.5,
            'microstructure': 0.3,
            'order_flow': 0.2
        }
        
        # Calculate weighted confidence
        combined_confidence = sum(
            signal.get('confidence', 0) * weight 
            for signal, weight in zip(
                [technical_signal, microstructure_signal, order_flow_signal],
                weights.values()
            )
        )
        
        # Determine action (majority vote)
        actions = [s.get('action', 'HOLD') for s in [technical_signal, microstructure_signal, order_flow_signal]]
        action_counts = {action: actions.count(action) for action in set(actions)}
        final_action = max(action_counts, key=action_counts.get)
        
        return {
            'action': final_action,
            'confidence': combined_confidence,
            'score': combined_confidence * 100,
            'strength': combined_confidence,
            'components': {
                'technical': technical_signal,
                'microstructure': microstructure_signal,
                'order_flow': order_flow_signal
            },
            'weights': weights,
            'timestamp': datetime.now()
        }
    
    def get_signal_statistics(self) -> Dict:
        """Get statistics about generated signals"""
        if not self.signal_history:
            return {'total_signals': 0}
        
        df = pd.DataFrame(self.signal_history)
        
        return {
            'total_signals': len(self.signal_history),
            'buy_signals': len(df[df['action'] == 'BUY']),
            'sell_signals': len(df[df['action'] == 'SELL']),
            'avg_confidence': df['confidence'].mean(),
            'avg_position_size': df['position_size'].mean(),
            'signals_per_hour': len(self.signal_history) / max(1, (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600)
        }

# Demo usage example
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'Open': 100 * (1 + np.random.randn(250).cumsum() * 0.02),
        'High': 100 * (1 + np.random.randn(250).cumsum() * 0.02 + 0.01),
        'Low': 100 * (1 + np.random.randn(250).cumsum() * 0.02 - 0.01),
        'Close': 100 * (1 + np.random.randn(250).cumsum() * 0.02),
        'Volume': np.random.randint(1000000, 5000000, 250)
    }, index=dates)
    
    # Initialize generator
    generator = SignalGenerator()
    
    # Generate signal
    signal = generator.generate_signal(sample_data)
    
    if signal:
        print("Signal Generated:")
        print(f"Action: {signal['action']}")
        print(f"Confidence: {signal['confidence']:.2%}")
        print(f"Position Size: {signal['position_size']:.2%}")
        print(f"Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"Take Profit: ${signal['take_profit']:.2f}")
        print(f"Reasons: {', '.join(signal['reasons'])}")