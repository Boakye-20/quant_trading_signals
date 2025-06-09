"""
Risk Manager - Public Demo Version
Basic risk management functionality for demonstration
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Basic risk management system for demonstration
    
    Production version includes:
    - Advanced portfolio optimization algorithms
    - Real-time VaR and CVaR calculations
    - Correlation matrix analysis
    - Stress testing scenarios
    - Dynamic position sizing based on market regime
    - Multi-factor risk models
    """
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize risk manager with basic parameters"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 0.1  # 10% max per position
        self.max_portfolio_risk = 0.2  # 20% max portfolio risk
        self.stop_loss_multiplier = 2.0  # 2x ATR
        self.take_profit_multiplier = 3.0  # 3x ATR
        
        # Track positions
        self.positions = {}
        self.position_count = 0
        
        logger.info(f"RiskManager initialized with capital: ${initial_capital:,.2f}")
    
    def check_signal(self, signal: Dict, current_price: float) -> Dict:
        """
        Validate trading signal against risk rules
        
        Returns risk-adjusted signal or rejection with reason
        """
        
        # Basic validation
        if not signal or signal.get('action') == 'HOLD':
            return {
                'approved': False,
                'reason': 'No actionable signal'
            }
        
        # Check position limits
        if signal['action'] == 'BUY' and self.position_count >= 5:
            return {
                'approved': False,
                'reason': 'Maximum positions reached',
                'max_positions': 5
            }
        
        # Calculate position size
        position_size = self._calculate_position_size(
            signal.get('confidence', 0.5),
            signal.get('volatility', 0.02)
        )
        
        # Check if we have enough capital
        position_value = position_size * self.current_capital
        if position_value > self.current_capital * 0.9:  # Keep 10% cash
            return {
                'approved': False,
                'reason': 'Insufficient capital',
                'required': position_value,
                'available': self.current_capital * 0.9
            }
        
        # Add risk parameters to signal
        signal['position_size'] = position_size
        signal['position_value'] = position_value
        signal['stop_loss'] = self._calculate_stop_loss(
            current_price, 
            signal['action'],
            signal.get('volatility', 0.02)
        )
        signal['take_profit'] = self._calculate_take_profit(
            current_price,
            signal['action'], 
            signal.get('volatility', 0.02)
        )
        
        return {
            'approved': True,
            'signal': signal,
            'risk_metrics': self.get_portfolio_metrics()
        }
    
    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """
        Calculate position size based on confidence and volatility
        
        Simplified Kelly Criterion implementation
        """
        # Base position size
        base_size = self.max_position_size
        
        # Adjust for confidence
        confidence_factor = min(1.0, confidence * 1.5)
        
        # Adjust for volatility (inverse relationship)
        volatility_factor = min(1.0, 0.02 / volatility)
        
        # Final position size
        position_size = base_size * confidence_factor * volatility_factor
        
        # Ensure within limits
        return min(self.max_position_size, max(0.01, position_size))
    
    def _calculate_stop_loss(self, price: float, action: str, volatility: float) -> float:
        """Calculate stop loss price"""
        stop_distance = price * volatility * self.stop_loss_multiplier
        
        if action == 'BUY':
            return round(price - stop_distance, 2)
        else:  # SELL
            return round(price + stop_distance, 2)
    
    def _calculate_take_profit(self, price: float, action: str, volatility: float) -> float:
        """Calculate take profit price"""
        profit_distance = price * volatility * self.take_profit_multiplier
        
        if action == 'BUY':
            return round(price + profit_distance, 2)
        else:  # SELL
            return round(price - profit_distance, 2)
    
    def get_portfolio_metrics(self) -> Dict:
        """
        Get current portfolio risk metrics
        
        Production version includes:
        - Real-time P&L tracking
        - Sharpe ratio calculation
        - Maximum drawdown analysis
        - Correlation metrics
        - Greeks for options positions
        """
        
        # Calculate basic metrics
        total_exposure = sum(
            pos.get('value', 0) for pos in self.positions.values()
        )
        
        cash_available = self.current_capital - total_exposure
        exposure_percentage = (total_exposure / self.initial_capital) * 100
        
        # Simple risk assessment
        if exposure_percentage > 80:
            risk_status = 'HIGH'
        elif exposure_percentage > 50:
            risk_status = 'MEDIUM'
        else:
            risk_status = 'LOW'
        
        return {
            'portfolio_value': self.current_capital,
            'cash_available': cash_available,
            'total_exposure': total_exposure,
            'exposure_percentage': exposure_percentage,
            'position_count': self.position_count,
            'max_position_size': self.max_position_size,
            'risk_status': risk_status,
            'last_updated': datetime.now().isoformat()
        }
    
    def update_position(self, symbol: str, action: str, price: float, size: float):
        """Update position tracking"""
        if action == 'BUY':
            self.positions[symbol] = {
                'entry_price': price,
                'size': size,
                'value': price * size * self.current_capital,
                'entry_time': datetime.now()
            }
            self.position_count += 1
        elif action == 'SELL' and symbol in self.positions:
            del self.positions[symbol]
            self.position_count -= 1
    
    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (simplified demo version)
        
        Production version uses:
        - Historical simulation
        - Monte Carlo simulation
        - Parametric VaR with full covariance matrix
        """
        
        # Simplified VaR calculation
        if not self.positions:
            return 0.0
        
        # Assume portfolio volatility of 15% annually
        portfolio_volatility = 0.15
        
        # Daily VaR
        daily_vol = portfolio_volatility / np.sqrt(252)
        
        # Z-score for confidence level
        if confidence_level == 0.95:
            z_score = 1.645
        elif confidence_level == 0.99:
            z_score = 2.326
        else:
            z_score = 1.645
        
        # VaR calculation
        portfolio_value = self.current_capital
        var = portfolio_value * daily_vol * z_score
        
        return var