import time
import requests
from typing import Dict, Any, List, Optional
from functools import wraps
import math
from binance.client import Client
from config import settings
from utils.logger import get_logger
from config.trading_pairs import TRADING_PAIRS

logger = get_logger(__name__)

def retry_on_exception(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
                    else:
                        logger.warning(f"Retrying {func.__name__} due to error: {e}")
                        time.sleep(1)  # Wait before retrying
        return wrapper
    return decorator

class BinanceClientWrapper:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self.request_weight = 0
        self.last_request_time = 0
        self.server_time_buffer = 2000  # 2 second buffer

    def initialize(self):
        self.client = Client(self.api_key, self.api_secret, testnet=settings.IS_TESTNET)

    def _get_server_time(self) -> int:
        """Get current server time with retries"""
        for attempt in range(3):
            try:
                return self.client.get_server_time()['serverTime']
            except Exception as e:
                logger.warning(f"Server time fetch attempt {attempt + 1} failed: {e}")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(1)
        raise Exception("Failed to get server time after 3 attempts")

    def _prepare_request(self, params: dict = None) -> dict:
        """Prepare request parameters with server-synchronized timestamp"""
        params = params or {}
        
        try:
            # Get server time first
            server_time = self._get_server_time()
            
            # Use server time minus buffer to ensure we're behind
            params['timestamp'] = server_time - self.server_time_buffer
            
            logger.debug(f"Request prepared with timestamp {params['timestamp']} (server time: {server_time})")
            return params
        except Exception as e:
            logger.error(f"Error preparing request parameters: {e}")
            raise

    def _wait_for_rate_limit(self):
        current_time = time.time()
        if current_time - self.last_request_time < 60:
            if self.request_weight > 1000:  # Weight limit per minute
                sleep_time = 60 - (current_time - self.last_request_time)
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_weight = 0
                self.last_request_time = time.time()
        else:
            self.request_weight = 0
            self.last_request_time = current_time

    @retry_on_exception()
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._wait_for_rate_limit()
        self.request_weight += 10
        exchange_info = self.client.get_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            if not any(f['filterType'] == 'MIN_NOTIONAL' for f in symbol_info['filters']):
                symbol_info['filters'].append({
                    'filterType': 'MIN_NOTIONAL',
                    'minNotional': '10.0'
                })
        return symbol_info

    @retry_on_exception()
    def get_klines(self, symbol: str, interval: str, limit: int = 10000) -> List[List]:
        self._wait_for_rate_limit()
        self.request_weight += 1
        return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

    @retry_on_exception()
    def get_symbol_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._wait_for_rate_limit()
        self.request_weight += 1
        return self.client.get_symbol_ticker(symbol=symbol)

    @retry_on_exception()
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        self._wait_for_rate_limit()
        self.request_weight += 1
        return self.client.get_order_book(symbol=symbol, limit=limit)

    @retry_on_exception()
    def get_account(self) -> Dict[str, Any]:
        """Get account information with server-synchronized timestamp"""
        self._wait_for_rate_limit()
        self.request_weight += 5
        
        try:
            params = self._prepare_request()
            return self.client.get_account(**params)
        except Exception as e:
            logger.error(f"Error getting account information: {e}")
            raise

    @retry_on_exception()
    def create_order(self, **params):
        """Create order with server-synchronized timestamp"""
        self._wait_for_rate_limit()
        self.request_weight += 1
        
        try:
            params = self._prepare_request(params)
            
            if 'quantity' in params:
                symbol = params['symbol']
                pair_config = self.get_pair_config(symbol)
                step_size = float(pair_config['step_size'])
                quantity_precision = pair_config['quantity_precision']
                quantity = float(params['quantity'])
                quantity = math.floor(quantity / step_size) * step_size
                params['quantity'] = f"{quantity:.{quantity_precision}f}"

            logger.info(f"Creating order with params: {params}")
            return self.client.create_order(**params)
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    @retry_on_exception()
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        self._wait_for_rate_limit()
        self.request_weight += 1
        return self.client.cancel_order(
            symbol=symbol,
            orderId=order_id,
            timestamp=self._get_server_time() - self.server_time_buffer
        )

    @retry_on_exception()
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        self._wait_for_rate_limit()
        self.request_weight += 1
        return self.client.get_open_orders(
            symbol=symbol,
            timestamp=self._get_server_time() - self.server_time_buffer
        )

    @retry_on_exception()
    def get_all_orders(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        self._wait_for_rate_limit()
        self.request_weight += 5
        return self.client.get_all_orders(
            symbol=symbol,
            limit=limit,
            timestamp=self._get_server_time() - self.server_time_buffer
        )

    @retry_on_exception()
    def get_my_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        self._wait_for_rate_limit()
        self.request_weight += 5
        return self.client.get_my_trades(
            symbol=symbol,
            limit=limit,
            timestamp=self._get_server_time() - self.server_time_buffer
        )

    def close_connection(self):
        # No need to close connection for requests-based client
        pass

    def get_total_balance_in_usdt(self) -> float:
        try:
            account_info = self.get_account()
            total_balance = 0.0

            for balance in account_info['balances']:
                asset = balance['asset']
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_asset_balance = free_balance + locked_balance

                if total_asset_balance > 0:
                    if asset == 'USDT':
                        total_balance += total_asset_balance
                    else:
                        symbol = f"{asset}USDT"
                        ticker = self.get_symbol_ticker(symbol)
                        if ticker:
                            price = float(ticker['price'])
                            asset_value_in_usdt = total_asset_balance * price
                            total_balance += asset_value_in_usdt
                        else:
                            logger.warning(f"Skipping {asset} due to invalid symbol or unavailable price")

            return total_balance

        except Exception as e:
            logger.error(f"Error getting total balance in USDT: {e}")
            return 0.0

    def get_pair_config(self, symbol: str) -> Dict[str, Any]:
        return TRADING_PAIRS.get(symbol, {})

    def get_account_balance(self):
        try:
            account_info = self.get_account()
            usdt_balance = next((float(balance['free']) for balance in account_info['balances'] if balance['asset'] == 'USDT'), 0)
            return usdt_balance
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0

    def create_market_order(self, symbol: str, side: str, quantity: float):
        try:
            order = self.create_order(
                symbol=symbol,
                side=side.upper(),
                type='MARKET',
                quantity=quantity
            )
            return order
        except Exception as e:
            logger.error(f"Unexpected error creating market order: {e}")
            return None

    def get_server_time(self):
        return self._get_server_time()

    def get_exchange_info(self):
        if not self.client:
            raise Exception("BinanceClientWrapper not initialized. Call initialize() first.")
        return self.client.get_exchange_info()

    def get_total_balance_in_usdt(self) -> float:
        try:
            account_info = self.get_account()
            total_balance = 0.0

            for balance in account_info['balances']:
                asset = balance['asset']
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_asset_balance = free_balance + locked_balance

                if total_asset_balance > 0:
                    if asset == 'USDT':
                        total_balance += total_asset_balance
                    else:
                        symbol = f"{asset}USDT"
                        ticker = self.get_symbol_ticker(symbol)
                        if ticker:
                            price = float(ticker['price'])
                            asset_value_in_usdt = total_asset_balance * price
                            total_balance += asset_value_in_usdt
                        else:
                            logger.warning(f"Skipping {asset} due to invalid symbol or unavailable price")

            return total_balance

        except Exception as e:
            logger.error(f"Error getting total balance in USDT: {e}")
            return 0.0

    def convert_all_assets_to_usdt(self):
        try:
            account_info = self.get_account()
            total_converted = 0.0

            for balance in account_info['balances']:
                asset = balance['asset']
                free_balance = float(balance['free'])

                if free_balance > 0 and asset != 'USDT':
                    symbol = f"{asset}USDT"
                    try:
                        # Get the current market price
                        ticker = self.get_symbol_ticker(symbol)
                        if ticker:
                            price = float(ticker['price'])
                            
                            # Calculate the quantity to sell (considering minimum notional value)
                            min_notional = self.get_min_notional(symbol)
                            quantity = max(free_balance, min_notional / price)
                            
                            # Create a market sell order
                            order = self.create_market_order(symbol, 'SELL', quantity)
                            
                            if order and order['status'] == 'FILLED':
                                converted_amount = float(order['cummulativeQuoteQty'])
                                total_converted += converted_amount
                                logger.info(f"Converted {quantity} {asset} to {converted_amount} USDT")
                            else:
                                logger.warning(f"Failed to convert {asset} to USDT")
                        else:
                            logger.warning(f"No ticker found for {symbol}, skipping conversion")
                    except Exception as e:
                        logger.error(f"Unexpected error converting {asset} to USDT: {e}")

            logger.info(f"Total converted to USDT: {total_converted}")
            return total_converted

        except Exception as e:
            logger.error(f"Error in convert_all_assets_to_usdt: {e}")
            return 0.0

    def get_min_notional(self, symbol: str) -> float:
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
                if min_notional_filter:
                    return float(min_notional_filter['minNotional'])
            return 10.0  # Default value if not found
        except Exception as e:
            logger.error(f"Error getting min notional for {symbol}: {e}")
            return 10.0  # Default value in case of error

    def get_top_trading_pairs(self, base_asset: str = 'USDT', limit: int = 20) -> List[str]:
        """
        Fetches the top trading pairs based on volume for a given base asset.
        
        :param base_asset: The base asset to filter trading pairs (e.g., 'USDT').
        :param limit: The number of top trading pairs to return.
        :return: A list of trading pair symbols.
        """
        try:
            self._wait_for_rate_limit()
            self.request_weight += 1
            exchange_info = self.client.get_exchange_info()
            symbols = exchange_info['symbols']
            
            # Filter symbols with the specified base asset and sort by volume
            filtered_symbols = [
                symbol['symbol'] for symbol in symbols
                if symbol['quoteAsset'] == base_asset and symbol['status'] == 'TRADING'
            ]
            
            # Fetch tickers to get volume information
            tickers = self.client.get_ticker()
            volume_dict = {ticker['symbol']: float(ticker['quoteVolume']) for ticker in tickers}
            
            # Sort symbols by volume and return the top 'limit' symbols
            top_symbols = sorted(filtered_symbols, key=lambda s: volume_dict.get(s, 0), reverse=True)[:limit]
            return top_symbols
        except Exception as e:
            logger.error(f"Error fetching top trading pairs: {e}")
            return []

# Usage example:
"""
async def main():
    client = BinanceClientWrapper(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
    
    # Get exchange info
    exchange_info = await client.get_exchange_info()
    print(f"Exchange info: {exchange_info}")
    
    # Get BTCUSDT ticker
    btc_ticker = await client.get_ticker('BTCUSDT')
    print(f"BTCUSDT ticker: {btc_ticker}")
    
    # Get account info
    account_info = await client.get_account()
    print(f"Account info: {account_info}")

if __name__ == "__main__":
    asyncio.run(main())

"""
