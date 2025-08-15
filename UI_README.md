# Predicto Bot Web UI

A modern, real-time web dashboard for monitoring and controlling your Polymarket trading bot.

## 🌟 Features

### Dashboard Overview
- **Real-time Status Monitoring**: Live connection status and bot state
- **Performance Metrics**: Total trades, P&L, active positions, and open orders
- **Market Data**: Live market information and orderbook data
- **Trade History**: Real-time trade execution updates

### Control Panel
- **Start/Stop Bot**: Remote control of bot execution
- **Real-time Updates**: WebSocket-powered live data streaming
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Modern UI
- **Beautiful Design**: Gradient backgrounds and glass-morphism effects
- **Dark/Light Theme**: Modern styling with excellent UX
- **Interactive Elements**: Hover effects and smooth animations
- **Toast Notifications**: Real-time alerts for bot actions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Run the Bot with UI
```bash
python run_bot.py
```

### 4. Access Dashboard
Open your browser to: **http://localhost:8000**

## 📊 Dashboard Components

### Header
- **Bot Status Indicator**: Shows connection and running state
- **Real-time Connection**: WebSocket status indicator

### Control Panel
- **Start Bot**: Begin trading operations
- **Stop Bot**: Halt all trading activities  
- **Refresh Data**: Manually update all data

### Statistics Overview
- **Total Trades**: Number of executed trades
- **P&L**: Current profit/loss
- **Active Positions**: Number of open positions
- **Open Orders**: Number of pending orders

### Data Tables
- **Markets**: Available trading markets with volume and pricing
- **Positions**: Current active positions with P&L
- **Orders**: Open orders with cancel functionality

## 🔧 API Endpoints

### REST API
- `GET /` - Dashboard page
- `GET /api/status` - Bot status
- `POST /api/start` - Start bot
- `POST /api/stop` - Stop bot
- `GET /api/markets` - Get markets
- `GET /api/positions` - Get positions
- `GET /api/orders` - Get orders

### WebSocket
- `WS /ws` - Real-time data streaming

## 🏗️ Architecture

### Backend (FastAPI)
```
src/ui/web_server.py - Main web server with WebSocket support
```

### Frontend (HTML/CSS/JS)
```
src/ui/templates/dashboard.html - Main dashboard template
src/ui/static/css/dashboard.css - Modern styling
src/ui/static/js/dashboard.js - Interactive functionality
```

### Integration
```
main.py - Updated to run web server alongside bot
src/strategies/base_strategy.py - Integrated with web server updates
```

## 🎨 UI Features

### Real-time Updates
- **WebSocket Connection**: Live data streaming
- **Auto-refresh**: Periodic data updates
- **Connection Recovery**: Automatic reconnection on disconnect

### Interactive Elements
- **Responsive Tables**: Sortable and scrollable data tables
- **Action Buttons**: Start/stop/refresh controls
- **Toast Notifications**: Success/error/warning alerts
- **Status Indicators**: Visual connection and bot state

### Modern Styling
- **Glass Morphism**: Semi-transparent cards with backdrop blur
- **Gradient Backgrounds**: Beautiful color transitions
- **Smooth Animations**: Hover effects and transitions
- **Mobile Responsive**: Works on all screen sizes

## 🔒 Security Features

### Test Mode
- **Safe Testing**: All operations run in test mode by default
- **No Real Trading**: Test mode prevents actual trades
- **Development Safety**: Perfect for development and testing

### Environment Configuration
- **Secure Credentials**: Private keys stored in environment variables
- **Configurable Settings**: All settings via .env file
- **Production Ready**: Easy switch to live trading

## 🛠️ Development

### File Structure
```
src/ui/
├── __init__.py
├── web_server.py          # FastAPI backend
├── templates/
│   └── dashboard.html     # Main dashboard
└── static/
    ├── css/
    │   └── dashboard.css  # Styling
    └── js/
        └── dashboard.js   # Frontend logic
```

### Adding Features
1. **Backend**: Add routes in `web_server.py`
2. **Frontend**: Update `dashboard.html` and `dashboard.js`
3. **Styling**: Modify `dashboard.css`

### Testing
```bash
python test_ui.py  # Test UI integration
```

## 🌐 Browser Support

- **Chrome/Edge**: Full support
- **Firefox**: Full support  
- **Safari**: Full support
- **Mobile Browsers**: Responsive design

## 📱 Mobile Experience

The dashboard is fully responsive and works great on:
- **Smartphones**: Optimized layout and touch interactions
- **Tablets**: Perfect for monitoring on larger mobile screens
- **Desktop**: Full-featured experience

## 🔄 Real-time Features

### WebSocket Updates
- **Bot Status**: Live running state
- **Trade Execution**: Real-time trade notifications
- **Market Data**: Live price and volume updates
- **Position Changes**: Instant position updates

### Auto-refresh
- **Smart Updates**: Only update when data changes
- **Efficient**: Minimal network usage
- **Reliable**: Handles connection issues gracefully

## 🎯 Use Cases

### Development
- **Test Strategies**: Safe environment for strategy testing
- **Debug Issues**: Real-time monitoring for debugging
- **Performance Analysis**: Live metrics and statistics

### Production
- **Monitor Trading**: Keep track of live trading operations
- **Quick Control**: Start/stop bot remotely
- **Performance Tracking**: Monitor P&L and trade statistics

### Analysis
- **Market Monitoring**: Track market conditions
- **Position Management**: Monitor active positions
- **Order Management**: Track and cancel orders

## 🚨 Important Notes

1. **Test Mode**: Bot runs in test mode by default for safety
2. **Private Keys**: Never commit private keys to version control
3. **Environment**: Always use .env file for configuration
4. **Security**: Use HTTPS in production environments
5. **Monitoring**: Keep an eye on the dashboard during trading

## 🎉 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Configure environment**: Copy and edit `.env.example`
4. **Run the bot**: `python run_bot.py`
5. **Open dashboard**: Visit `http://localhost:8000`
6. **Start trading**: Click the "Start Bot" button

The UI provides a complete interface for managing your Polymarket trading bot with real-time monitoring, control capabilities, and a beautiful modern design!
