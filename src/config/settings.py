"""
Configuration settings for the Polymarket trading bot.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Polymarket API Configuration
    polymarket_api_url: str = "https://clob.polymarket.com"
    polymarket_ws_url: str = "wss://clob.polymarket.com"
    
    # Wallet Configuration
    private_key: Optional[str] = None
    wallet_address: Optional[str] = None
    
    # Trading Configuration
    default_slippage: float = 0.01  # 1%
    max_position_size: float = 1000.0  # USD
    risk_per_trade: float = 0.02  # 2% of portfolio
    
    # Database Configuration
    database_url: str = "sqlite:///trades.db"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "bot.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
