// Dashboard JavaScript
class Dashboard {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
    }
    
    setupEventListeners() {
        // Control buttons
        document.getElementById('start-btn').addEventListener('click', () => this.startBot());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopBot());
        document.getElementById('refresh-btn').addEventListener('click', () => this.refreshData());
        
        // Handle page visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && !this.isConnected) {
                this.connectWebSocket();
            }
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected', 'Connected');
                this.showToast('Connected to bot', 'success');
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('disconnected', 'Disconnected');
                
                // Attempt to reconnect
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    setTimeout(() => this.connectWebSocket(), 3000);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showToast('Connection error', 'error');
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.showToast('Failed to connect', 'error');
        }
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'status':
                this.updateBotStatus(message.data);
                break;
            case 'trade':
                this.handleTradeUpdate(message.data);
                break;
            case 'error':
                this.showToast(message.data.message, 'error');
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    updateConnectionStatus(status, text) {
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        
        statusDot.className = `status-dot ${status}`;
        statusText.textContent = text;
    }
    
    updateBotStatus(status) {
        // Update status indicator
        if (status.running) {
            this.updateConnectionStatus('running', 'Bot Running');
        } else {
            this.updateConnectionStatus('connected', 'Bot Stopped');
        }
        
        // Update stats
        document.getElementById('total-trades').textContent = status.total_trades || 0;
        document.getElementById('pnl').textContent = this.formatCurrency(status.pnl || 0);
        document.getElementById('positions-count').textContent = (status.positions || []).length;
        document.getElementById('orders-count').textContent = (status.orders || []).length;
        
        // Update last update time
        if (status.last_update) {
            const lastUpdate = new Date(status.last_update).toLocaleString();
            document.getElementById('last-update').textContent = lastUpdate;
        }
        
        // Update button states
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        if (status.running) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
        
        // Update tables
        this.updateMarketsTable(status.markets || []);
        this.updatePositionsTable(status.positions || []);
        this.updateOrdersTable(status.orders || []);
    }
    
    updateMarketsTable(markets) {
        const tbody = document.querySelector('#markets-table tbody');
        
        if (markets.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="no-data">No markets loaded</td></tr>';
            return;
        }
        
        tbody.innerHTML = markets.map(market => `
            <tr>
                <td>${this.truncateText(market.question || 'Unknown Market', 50)}</td>
                <td>${this.formatCurrency(market.volume || 0)}</td>
                <td>${market.price ? '$' + market.price.toFixed(4) : 'N/A'}</td>
                <td><span class="status-badge status-active">Active</span></td>
            </tr>
        `).join('');
    }
    
    updatePositionsTable(positions) {
        const tbody = document.querySelector('#positions-table tbody');
        
        if (positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="no-data">No active positions</td></tr>';
            return;
        }
        
        tbody.innerHTML = positions.map(position => `
            <tr>
                <td>${this.truncateText(position.market_name || position.market_id, 30)}</td>
                <td><span class="side-${position.side.toLowerCase()}">${position.side}</span></td>
                <td>${position.size}</td>
                <td>${this.formatCurrency(position.entry_price || 0)}</td>
                <td class="${position.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                    ${this.formatCurrency(position.pnl || 0)}
                </td>
            </tr>
        `).join('');
    }
    
    updateOrdersTable(orders) {
        const tbody = document.querySelector('#orders-table tbody');
        
        if (orders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="no-data">No open orders</td></tr>';
            return;
        }
        
        tbody.innerHTML = orders.map(order => `
            <tr>
                <td>${this.truncateText(order.market_name || order.market_id, 30)}</td>
                <td><span class="side-${order.side.toLowerCase()}">${order.side}</span></td>
                <td>${order.size}</td>
                <td>${this.formatCurrency(order.price || 0)}</td>
                <td><span class="status-badge status-active">${order.status || 'Open'}</span></td>
                <td>
                    <button class="btn btn-danger btn-sm" onclick="dashboard.cancelOrder('${order.id}')">
                        Cancel
                    </button>
                </td>
            </tr>
        `).join('');
    }
    
    async startBot() {
        try {
            const response = await fetch('/api/start', { method: 'POST' });
            const result = await response.json();
            
            if (result.status === 'started') {
                this.showToast('Bot started successfully', 'success');
            } else {
                this.showToast('Bot is already running', 'warning');
            }
        } catch (error) {
            console.error('Error starting bot:', error);
            this.showToast('Failed to start bot', 'error');
        }
    }
    
    async stopBot() {
        try {
            const response = await fetch('/api/stop', { method: 'POST' });
            const result = await response.json();
            
            if (result.status === 'stopped') {
                this.showToast('Bot stopped successfully', 'success');
            } else {
                this.showToast('Bot is already stopped', 'warning');
            }
        } catch (error) {
            console.error('Error stopping bot:', error);
            this.showToast('Failed to stop bot', 'error');
        }
    }
    
    async refreshData() {
        try {
            // Refresh markets
            const marketsResponse = await fetch('/api/markets');
            const marketsData = await marketsResponse.json();
            
            // Refresh positions
            const positionsResponse = await fetch('/api/positions');
            const positionsData = await positionsResponse.json();
            
            // Refresh orders
            const ordersResponse = await fetch('/api/orders');
            const ordersData = await ordersResponse.json();
            
            // Update tables
            this.updateMarketsTable(marketsData.markets || []);
            this.updatePositionsTable(positionsData.positions || []);
            this.updateOrdersTable(ordersData.orders || []);
            
            this.showToast('Data refreshed', 'success');
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showToast('Failed to refresh data', 'error');
        }
    }
    
    async cancelOrder(orderId) {
        if (!confirm('Are you sure you want to cancel this order?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/orders/${orderId}`, { method: 'DELETE' });
            const result = await response.json();
            
            if (result.success) {
                this.showToast('Order cancelled successfully', 'success');
                this.refreshData();
            } else {
                this.showToast('Failed to cancel order', 'error');
            }
        } catch (error) {
            console.error('Error cancelling order:', error);
            this.showToast('Failed to cancel order', 'error');
        }
    }
    
    async loadInitialData() {
        // Load initial status
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            this.updateBotStatus(status);
        } catch (error) {
            console.error('Error loading initial status:', error);
        }
        
        // Load markets
        this.refreshData();
    }
    
    handleTradeUpdate(trade) {
        this.showToast(`Trade executed: ${trade.side} ${trade.size} @ ${trade.price}`, 'success');
        
        // Refresh data to show updated positions/orders
        setTimeout(() => this.refreshData(), 1000);
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        container.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
        
        // Make toast clickable to dismiss
        toast.addEventListener('click', () => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    }
    
    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 4
        }).format(amount);
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }
}

// Initialize dashboard when page loads
const dashboard = new Dashboard();
