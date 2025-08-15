"""
Trading Bot Configuration
Simple configuration system for live Polymarket trading.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TradingConfig:
    """Trading bot configuration."""
    
    # API Configuration
    private_key: Optional[str] = None
    polymarket_api_url: str = "https://clob.polymarket.com"
    test_mode: bool = True
    
    # Strategy Configuration
    use_enhanced_strategy: bool = True  # Use enhanced vs basic profitable strategy
    
    # Trading Parameters (adjusted for smaller bankrolls)
    base_position_size: float = 10.0   # Base position size in USD
    max_position_size: float = 20.0    # Maximum position size in USD
    max_positions: int = 3             # Maximum concurrent positions
    max_markets: int = 6               # Maximum markets to monitor
    
    # Risk Management
    stop_loss_pct: float = 0.04        # 4% stop loss
    take_profit_pct: float = 0.06      # 6% take profit
    max_daily_loss: float = 200.0      # Maximum daily loss in USD
    
    # Timing
    update_frequency_seconds: int = 300  # Check markets every 5 minutes
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        # Load from environment
        self.private_key = os.getenv('PRIVATE_KEY')
        self.test_mode = os.getenv('TEST_MODE', 'true').lower() == 'true'
        self.use_enhanced_strategy = os.getenv('USE_ENHANCED_STRATEGY', 'true').lower() == 'true'
        
        # Trading parameters from environment
        self.base_position_size = float(os.getenv('BASE_POSITION_SIZE', self.base_position_size))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', self.max_position_size))
        self.max_positions = int(os.getenv('MAX_POSITIONS', self.max_positions))
        self.max_markets = int(os.getenv('MAX_MARKETS', self.max_markets))
        
        # Risk management from environment
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PCT', self.stop_loss_pct))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PCT', self.take_profit_pct))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', self.max_daily_loss))
        
        # Timing from environment
        self.update_frequency_seconds = int(os.getenv('UPDATE_FREQUENCY_SECONDS', self.update_frequency_seconds))
        
        # Create .env file if it doesn't exist
        self._create_env_file_if_needed()
    
    def _create_env_file_if_needed(self):
        """Create a .env file with default values if it doesn't exist."""
        env_file = Path('.env')
        
        if not env_file.exists():
            env_content = """# Polymarket Trading Bot Configuration

# REQUIRED: Your Ethereum private key for trading
PRIVATE_KEY=your_private_key_here

# Trading Mode (true for test mode, false for live trading)
TEST_MODE=true

# Strategy Selection (true for enhanced, false for basic)
USE_ENHANCED_STRATEGY=true

# Position Sizing (in USD) - Adjusted for smaller bankrolls
BASE_POSITION_SIZE=10.0
MAX_POSITION_SIZE=20.0
MAX_POSITIONS=3
MAX_MARKETS=6

# Risk Management
STOP_LOSS_PCT=0.04
TAKE_PROFIT_PCT=0.06
MAX_DAILY_LOSS=200.0

# Update Frequency (seconds)
UPDATE_FREQUENCY_SECONDS=300
"""
            env_file.write_text(env_content)
            print(f"Created .env file at {env_file.absolute()}")
            print("Please edit the .env file and add your private key before running the bot.")
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy-specific configuration."""
        base_config = {
            # Core strategy parameters
            'base_position_size': self.base_position_size,
            'max_position_size': self.max_position_size,
            'max_positions': self.max_positions,
            
            # Risk management
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            
            # Crypto-optimized parameters [[memory:6262089]]
            'momentum_periods': 6,
            'momentum_threshold': 0.015,  # 1.5% momentum required
            'trend_confirmation_periods': 3,
            'confidence_multiplier': 1.5,
            
            # Profit targets
            'quick_profit_target': 0.03,     # 3% quick profit
            'extended_profit_target': 0.06,  # 6% extended profit
            
            # Trade timing
            'min_hold_time_minutes': 30,     # Minimum hold time
            'max_hold_time_hours': 4,        # Maximum hold time
        }
        
        if self.use_enhanced_strategy:
            # Enhanced strategy specific parameters
            base_config.update({
                'signal_strength_threshold': 0.7,
                'confirmation_indicators': 3,
                'noise_filter_periods': 5,
                'volatility_profit_multiplier': 1.5,
                'adaptive_parameters': True,
                'kelly_criterion_sizing': True,
            })
        
        return base_config
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.private_key or self.private_key == 'your_private_key_here':
            print("ERROR: Private key is required. Please set PRIVATE_KEY in .env file.")
            return False
        
        if self.base_position_size <= 0:
            print("ERROR: Base position size must be positive.")
            return False
        
        if self.max_position_size < self.base_position_size:
            print("ERROR: Max position size must be >= base position size.")
            return False
        
        if self.max_positions <= 0:
            print("ERROR: Max positions must be positive.")
            return False
        
        return True
    
    def display_config(self):
        """Display current configuration."""
        print("🔧 TRADING BOT CONFIGURATION")
        print("=" * 40)
        print(f"Test Mode: {self.test_mode}")
        print(f"Strategy: {'Enhanced Profitable' if self.use_enhanced_strategy else 'Basic Profitable'}")
        print(f"Base Position Size: ${self.base_position_size}")
        print(f"Max Position Size: ${self.max_position_size}")
        print(f"Max Positions: {self.max_positions}")
        print(f"Max Markets: {self.max_markets}")
        print(f"Stop Loss: {self.stop_loss_pct:.1%}")
        print(f"Take Profit: {self.take_profit_pct:.1%}")
        print(f"Update Frequency: {self.update_frequency_seconds}s")
        print(f"Private Key: {'✅ Set' if self.private_key and self.private_key != 'your_private_key_here' else '❌ Not Set'}")
        print("=" * 40)


if __name__ == "__main__":
    # Test configuration loading
    config = TradingConfig()
    config.display_config()
    
    if config.validate():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors")
