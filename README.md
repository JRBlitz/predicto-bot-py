# predicto-bot-py

A Python-based prediction bot for automated trading on Polymarket using official Polymarket repositories.

## Features

- **Automated Trading**: Place orders automatically based on configurable strategies
- **Mean Reversion Strategy**: Built-in strategy that trades when price deviates from moving average
- **Real-time Market Data**: Connect to Polymarket's CLOB for live market data
- **Risk Management**: Configurable position sizing and risk parameters
- **Extensible**: Easy to add new trading strategies

## Dependencies

This bot integrates with several official Polymarket repositories:

- [py-clob-client](https://github.com/Polymarket/py-clob-client) - Python client for Polymarket CLOB
- [python-order-utils](https://github.com/Polymarket/python-order-utils) - Order generation and signing utilities
- [poly-py-eip712-structs](https://github.com/Polymarket/poly-py-eip712-structs) - EIP712 data structure management

## Setup

1. **Clone and navigate to the project:**
   ```bash
   cd predicto-bot-py
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up your wallet:**
   - Add your private key to `.env`
   - Ensure your wallet has sufficient funds on Polygon network

## Configuration

Edit `.env` file with your settings:

```env
# Polymarket API Configuration
POLYMARKET_API_URL=https://clob.polymarket.com
POLYMARKET_WS_URL=wss://clob.polymarket.com

# Wallet Configuration
PRIVATE_KEY=your_private_key_here
WALLET_ADDRESS=your_wallet_address_here

# Trading Configuration
DEFAULT_SLIPPAGE=0.01  # 1%
MAX_POSITION_SIZE=1000  # USD
RISK_PER_TRADE=0.02    # 2% of portfolio
```

## Usage

Run the bot:

```bash
python main.py
```

The bot will:
1. Connect to Polymarket's CLOB
2. Monitor available markets
3. Execute trades based on the mean reversion strategy
4. Log all activities to `bot.log`

## Strategy Configuration

The mean reversion strategy parameters can be adjusted in `main.py`:

```python
strategy_config = {
    'lookback_period': 20,        # Period for moving average calculation
    'deviation_threshold': 0.05,  # 5% deviation threshold
    'position_size_pct': 0.1,     # 10% of portfolio per trade
    'base_position_size': 100,    # Base position size in USD
    'interval': 60               # Check every 60 seconds
}
```

## Adding New Strategies

1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement the required abstract methods
3. Add your strategy to the main bot

Example:
```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    async def analyze_market(self, market_data):
        # Your analysis logic
        pass
    
    async def should_enter_position(self, signals):
        # Your entry logic
        pass
    
    # ... implement other required methods
```

## Security Notes

- Never commit your `.env` file with private keys
- Use a dedicated trading wallet with limited funds
- Test with small amounts first
- Monitor the bot's performance regularly

## Disclaimer

This bot is for educational purposes. Trading involves risk of loss. Use at your own risk and never trade with money you cannot afford to lose.

## License

MIT License
