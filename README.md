# Polymarket Trading Bot

A streamlined cryptocurrency trading bot for Polymarket, focused on Bitcoin, Ethereum, XRP, and Solana markets.

## Features

- **Live Trading**: Real-time trading with Polymarket CLOB API
- **Crypto Focus**: Exclusively trades Bitcoin, Ethereum, XRP, and Solana markets
- **Profitable Strategies**: Two proven strategies (Basic and Enhanced Profitable)
- **Risk Management**: Built-in stop losses, position sizing, and daily limits
- **Simple Configuration**: Environment-based configuration system

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   # Edit .env file (created automatically on first run)
   nano .env
   
   # Add your Ethereum private key
   PRIVATE_KEY=your_private_key_here
   
   # Set to false for live trading (default is test mode)
   TEST_MODE=true
   ```

3. **Run the Bot**:
   ```bash
   python main.py
   ```

## Configuration

The bot uses environment variables for configuration. Key settings:

- `PRIVATE_KEY`: Your Ethereum private key (required)
- `TEST_MODE`: Set to `false` for live trading (default: `true`)
- `USE_ENHANCED_STRATEGY`: Use enhanced vs basic strategy (default: `true`)
- `BASE_POSITION_SIZE`: Base position size in USD (default: `100`)
- `MAX_POSITIONS`: Maximum concurrent positions (default: `3`)
- `STOP_LOSS_PCT`: Stop loss percentage (default: `0.04`)
- `TAKE_PROFIT_PCT`: Take profit percentage (default: `0.06`)

## API Key Setup

For live trading, you need to create API credentials on Polymarket:

1. Go to clob.polymarket.com
2. Sign EIP-712 ClobAuth with chainId 137
3. Send POST /auth/api-key with L1 headers
4. Save {key, secret, passphrase} into keychain under 'trading-api-creds'

## Safety Features

- **Test Mode**: Safe testing without real trades
- **Position Limits**: Maximum number of concurrent positions
- **Daily Loss Limits**: Automatic shutdown on daily loss threshold
- **Stop Losses**: Automatic position exits on adverse moves
- **Crypto Only**: Filters out non-crypto markets automatically

## Project Structure

```
predicto-bot-py/
├── main.py                          # Main entry point
├── config.py                        # Configuration system
├── crypto_market_config.py          # Crypto market definitions
├── requirements.txt                 # Dependencies
├── src/
│   ├── strategies/                  # Trading strategies
│   │   ├── base_strategy.py
│   │   ├── profitable_momentum_strategy.py
│   │   └── enhanced_profitable_strategy.py
│   ├── utils/
│   │   └── polymarket_client.py     # Polymarket API client
│   ├── backtesting/                 # Strategy validation
│   └── config/
└── logs/                           # Trading logs
```

## Supported Markets

The bot focuses on crypto markets across multiple timeframes:

- **Bitcoin**: Hourly, daily, weekly, monthly markets
- **Ethereum**: Hourly, daily, weekly, monthly markets  
- **XRP**: Hourly, daily, weekly, monthly markets
- **Solana**: Hourly, daily, weekly, monthly markets

## Risk Warning

This bot trades with real money when TEST_MODE=false. Always:

- Start with TEST_MODE=true
- Use small position sizes initially
- Monitor performance closely
- Never risk more than you can afford to lose

## Support

For issues or questions, check the logs in the `logs/` directory for detailed information about bot operations.
