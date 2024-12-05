import asyncio
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from config import settings
from config.trading_pairs import get_all_symbols, update_trading_pairs
from models.trade import Trade, TradeStatus
from indicators.bollinger_bands import BollingerBands
from data.market_data import MarketData
from exchange.binance_client import BinanceClientWrapper
from utils.logger import get_logger
from utils.risk_management import RiskManager
import pandas_ta as ta
import math
from datetime import datetime
from utils.performance_tracker import PerformanceTracker
from ml.price_predictor import PricePredictor
import time
from cachetools import TTLCache
from typing import Tuple
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = get_logger(__name__)

class ScalpingStrategy:
    def __init__(self, market_data: MarketData, exchange_client: BinanceClientWrapper, risk_manager: RiskManager):
        self.market_data = market_data
        self.exchange_client = exchange_client
        self.risk_manager = risk_manager
        self.bb_indicator = BollingerBands()
        self.open_trades = {}
        self.trade_history = []
        self.bb_period = 20
        self.stop_loss_percentage = 0.015
        self.mse_threshold = 15
        self.take_profit_percentage = 0.025
        self.performance_tracker = PerformanceTracker()
        self.price_predictor = PricePredictor()
        self.symbols = get_all_symbols()
        self.timeframe = settings.TIMEFRAME
        self.prediction_threshold = 0.005
        self.model_trained = False
        self.training_interval = 24 * 60 * 60
        self.last_model_training = 0
        self.model_file = 'price_predictor_model.joblib'
        self.last_trade_attempt = {}
        self.cool_down_period = 30
        self.max_trade_duration = 1800
        self.historical_data_cache = TTLCache(maxsize=100, ttl=3600)
        self.open_trades: Dict[str, Trade] = {}
        self.max_position_size = 0.03
        self.trade_history: List[Trade] = []
        self.trailing_stop_percentage = 0.008
        self.volatility_window = 20
        self.risk_per_trade = 0.015
        self.bb = BollingerBands(window=20, num_std=2)
        self.last_model_training_time = 0
        self.last_pairs_update_time = 0
        self.atr_period = 14
        self.position_size_atr_multiplier = 2.0
        self.max_position_size_usd = 5000  # Maximum position size in USD
        self.min_position_size_usd = 100   # Minimum position size in USD

    async def warm_up(self):
        logger.info("Starting warm-up period...")
        await self.train_model()
        for symbol in self.symbols:
            await self.market_data.get_historical_data(symbol, self.timeframe, limit=10000)
        logger.info("Warm-up period completed")

    async def run_iteration(self):
        await self.warm_up()
        try:
            logger.info("Running strategy iteration")
            account_balance = self.exchange_client.get_account_balance()
            logger.info(f"Current USDT balance: ${account_balance}")
            await self.update_trading_pairs()
            current_time = time.time()        
            if current_time - self.last_model_training_time > 3600: #Train Model after every 1hr
                self.last_pairs_update_time = current_time
                await self.train_model()
                self.last_model_training_time = current_time

            new_trades = []
            await self.update_open_trades()
            await self.manage_open_trades()
            
            for symbol in self.symbols:
                try:
                    if symbol in self.last_trade_attempt and time.time() - self.last_trade_attempt[symbol] < self.cool_down_period:
                        logger.info(f"Skipping {symbol} due to active cool-down period")
                        continue

                    signal = await self.generate_trading_signal(symbol)
                    if signal:
                        trade = await self.execute_trade(symbol, signal)
                        if trade:
                            new_trades.append(trade)
                            self.open_trades[symbol] = trade
                            self.last_trade_attempt[symbol] = time.time()
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)

            self.log_open_trades()
            current_balance = await self.get_account_balance()
            self.performance_tracker.update_balance(current_balance)

            return new_trades, current_balance
        except Exception as e:
            logger.error(f"Error in run_iteration: {str(e)}", exc_info=True)
            return [], await self.get_account_balance()

    async def generate_trading_signal(self, symbol: str) -> Optional[str]:
        logger.info(f"Generating trading signal for {symbol}")
        historical_data = await self.market_data.get_historical_data(symbol, self.timeframe, limit=10000)
        if historical_data is None or len(historical_data) < 1000:
            logger.warning(f"Not enough historical data for {symbol}")
            return None

        current_price = historical_data['close'].iloc[-1]
        
        # Calculate indicators
        close_prices = historical_data['close']
        high_prices = historical_data['high']
        low_prices = historical_data['low']
        volume = historical_data['volume']
        
        # MACD with faster settings
        macd = ta.macd(close_prices, fast=6, slow=13, signal=4)
        macd_line = macd['MACD_6_13_4']
        signal_line = macd['MACDs_6_13_4']
        
        # RSI with shorter period
        rsi = ta.rsi(close_prices, length=7)
        
        # Bollinger Bands with shorter period
        bb_result = self.bb_indicator.calculate(close_prices)
        
        # Calculate price change prediction
        features = self.prepare_features(historical_data)
        price_change_prediction = self.price_predictor.predict(features[-1].reshape(1, -1))[0]
        
        # Calculate additional indicators
        adx = ta.adx(high_prices, low_prices, close_prices, length=7)['ADX_7']
        
        # Get dynamic thresholds
        thresholds = await self.calculate_dynamic_thresholds(symbol)
        if thresholds is None:
            logger.warning(f"Using default thresholds for {symbol}")
            thresholds = {
                'macd_threshold': abs(macd_line.iloc[-1] - signal_line.iloc[-1]) * 0.2,  # More conservative
                'bb_threshold': (bb_result['upper'] - bb_result['lower']) * 0.2,  # More conservative
                'rsi_overbought': 70,  # More conservative
                'rsi_oversold': 30,  # More conservative
                'adx_threshold': 25,  # Stronger trend requirement
                'volume_threshold': volume.rolling(window=20).mean().iloc[-1] * 2.0,  # Higher volume requirement
                'min_signal_strength': 0.8,
                'max_signal_strength': 1.2
            }
        
        logger.info(f"Indicators for {symbol}: MACD={macd_line.iloc[-1]:.4f}, Signal={signal_line.iloc[-1]:.4f}, "
                    f"RSI={rsi.iloc[-1]:.2f}, BB Upper={bb_result['upper']:.4f}, "
                    f"BB Lower={bb_result['lower']:.4f}, Price={current_price:.4f}, "
                    f"Prediction={price_change_prediction:.4f}, ADX={adx.iloc[-1]:.2f}")

        # Enhanced buy conditions with stricter confirmation requirements
        buy_conditions = [
            macd_line.iloc[-1] > signal_line.iloc[-1],  # MACD crossover
            macd_line.iloc[-1] - signal_line.iloc[-1] > thresholds['macd_threshold'],  # Strong MACD momentum
            current_price > bb_result['lower'],  # Price above lower band
            rsi.iloc[-1] < thresholds['rsi_overbought'],  # Not overbought
            rsi.iloc[-1] > thresholds['rsi_oversold'],  # Not oversold
            price_change_prediction > current_price * 1.005,  # Minimum 0.5% upside predicted
            adx.iloc[-1] > thresholds['adx_threshold'],  # Strong trend
            volume.iloc[-1] > thresholds['volume_threshold'],  # High volume
            current_price > bb_result['middle']  # Price above middle band
        ]

        # Enhanced sell conditions with stricter confirmation requirements
        sell_conditions = [
            macd_line.iloc[-1] < signal_line.iloc[-1],  # MACD crossover
            signal_line.iloc[-1] - macd_line.iloc[-1] > thresholds['macd_threshold'],  # Strong MACD momentum
            current_price < bb_result['upper'],  # Price below upper band
            rsi.iloc[-1] > thresholds['rsi_oversold'],  # Not oversold
            rsi.iloc[-1] < thresholds['rsi_overbought'],  # Not overbought
            price_change_prediction < current_price * 0.995,  # Minimum 0.5% downside predicted
            adx.iloc[-1] > thresholds['adx_threshold'],  # Strong trend
            volume.iloc[-1] > thresholds['volume_threshold'],  # High volume
            current_price < bb_result['middle']  # Price below middle band
        ]

        trend = await self.get_trend(symbol)
        account_balance = self.exchange_client.get_account_balance()
        volatility = await self.calculate_volatility(symbol)

        # Calculate signal strength with normalization
        signal_strength = sum([
            1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1,
            1 if current_price > bb_result['middle'] else -1,
            1 if rsi.iloc[-1] > 50 else -1,
            1 if adx.iloc[-1] > thresholds['adx_threshold'] else -1,
            1 if volume.iloc[-1] > thresholds['volume_threshold'] else -1
        ]) / 5.0  # Normalize to [-1, 1] range
        
        signal_strength = (signal_strength + 1) / 2  # Convert to [0, 1] range
        signal_strength = max(min(signal_strength, thresholds['max_signal_strength']), thresholds['min_signal_strength'])

        # Enhanced signal generation with stronger confirmation
        if sum(buy_conditions) >= 7 and trend in ['up', 'strong_up']:  # At least 7 conditions must be met
            if trend == 'strong_up':
                signal_strength *= 1.3  # Increased weight for strong trends
            
            position_size = await self.calculate_position_size(symbol, account_balance, volatility, current_price)
            
            # Log detailed buy conditions
            logger.info(f"Buy signal analysis for {symbol}:")
            logger.info(f"MACD Momentum: {macd_line.iloc[-1] - signal_line.iloc[-1]:.8f}")
            logger.info(f"RSI: {rsi.iloc[-1]:.2f}")
            logger.info(f"ADX: {adx.iloc[-1]:.2f}")
            logger.info(f"Predicted Upside: {((price_change_prediction/current_price - 1) * 100):.2f}%")
            logger.info(f"Volume Ratio: {(volume.iloc[-1]/thresholds['volume_threshold']):.2f}")
            logger.info(f"Signal Strength: {signal_strength:.2f}")
            
            if position_size > 0:
                return ('buy', position_size)
            else:
                logger.warning(f"Buy signal generated for {symbol} but position size is invalid: {position_size}")
                return None
                
        elif sum(sell_conditions) >= 7 and trend in ['down', 'strong_down']:  # At least 7 conditions must be met
            if trend == 'strong_down':
                signal_strength *= 1.3  # Increased weight for strong trends
            
            position_size = await self.calculate_position_size(symbol, account_balance, volatility, current_price)
            
            # Log detailed sell conditions
            logger.info(f"Sell signal analysis for {symbol}:")
            logger.info(f"MACD Momentum: {signal_line.iloc[-1] - macd_line.iloc[-1]:.8f}")
            logger.info(f"RSI: {rsi.iloc[-1]:.2f}")
            logger.info(f"ADX: {adx.iloc[-1]:.2f}")
            logger.info(f"Predicted Downside: {((price_change_prediction/current_price - 1) * 100):.2f}%")
            logger.info(f"Volume Ratio: {(volume.iloc[-1]/thresholds['volume_threshold']):.2f}")
            logger.info(f"Signal Strength: {signal_strength:.2f}")
            
            if position_size > 0:
                return ('sell', position_size)
            else:
                logger.warning(f"Sell signal generated for {symbol} but position size is invalid: {position_size}")
                return None
        
        # Enhanced logging for signal analysis
        logger.debug(f"Signal analysis for {symbol}:")
        logger.debug(f"Buy conditions met: {sum(buy_conditions)}/{len(buy_conditions)}")
        logger.debug(f"Sell conditions met: {sum(sell_conditions)}/{len(sell_conditions)}")
        logger.debug(f"Current trend: {trend}")
        logger.debug(f"MACD Momentum: {macd_line.iloc[-1] - signal_line.iloc[-1]:.8f}")
        logger.debug(f"RSI: {rsi.iloc[-1]:.2f}")
        logger.debug(f"ADX: {adx.iloc[-1]:.2f}")
        logger.debug(f"Price vs BB Middle: {((current_price/bb_result['middle'] - 1) * 100):.2f}%")
        
        return None

    async def update_trading_pairs(self):
        try:
            top_pairs = self.exchange_client.get_top_trading_pairs(base_asset='USDT', limit=20)
            if top_pairs:
                update_trading_pairs(top_pairs)
                self.symbols = top_pairs
                logger.info(f"Updated trading pairs: {top_pairs}")
            else:
                logger.warning("Failed to fetch top trading pairs. Trading pairs remain unchanged.")
        except Exception as e:
            logger.error(f"Error updating trading pairs: {e}")

    def calculate_macd(self, prices, fast=7, slow=15, signal=5):
        if len(prices) < max(fast, slow, signal):
            logger.warning(f"Not enough data points to calculate MACD. Required: {max(fast, slow, signal)}, Available: {len(prices)}")
            return None, None, None
        
        prices_series = pd.Series(prices)
        ema_fast = ta.ema(prices_series, length=fast)
        ema_slow = ta.ema(prices_series, length=slow)
        
        if ema_fast is None or ema_slow is None:
            logger.warning("EMA calculation failed")
            return None, None, None
        
        macd = ema_fast - ema_slow
        signal_line = ta.ema(macd, length=signal)
        
        if signal_line is None:
            logger.warning("Signal line calculation failed")
            return None, None, None
        
        # Ensure all arrays have the same length
        min_length = min(len(macd), len(signal_line))
        macd = macd[-min_length:].reset_index(drop=True)
        signal_line = signal_line[-min_length:].reset_index(drop=True)
        
        logger.debug(f"MACD calculation result: MACD length: {len(macd)}, Signal length: {len(signal_line)}")
        logger.debug(f"MACD last 5 values: {macd.tail()}")
        logger.debug(f"Signal line last 5 values: {signal_line.tail()}")
        
        return macd, signal_line, macd - signal_line

    async def fallback_strategy(self, symbol: str) -> Optional[str]:
        try:
            data = await self.market_data.get_historical_data(symbol, self.timeframe, limit=10000)
            if data is None or len(data) < 1000:
                return None

            close_prices = data['close'].values
            sma_20 = np.mean(close_prices[-20:])
            sma_50 = np.mean(close_prices[-50:])
            
            if close_prices[-1] > sma_20 and sma_20 > sma_50:
                return 'buy'
            elif close_prices[-1] < sma_20 and sma_20 < sma_50:
                return 'sell'
            
            return None
        except Exception as e:
            logger.error(f"Error in fallback strategy for {symbol}: {str(e)}", exc_info=True)
            return None

    async def calculate_dynamic_thresholds(self, symbol: str) -> Dict[str, float]:
        try:
            historical_data = await self.market_data.get_historical_data(symbol, self.timeframe, limit=100)
            if historical_data is None:
                return None

            # Calculate price-based scaling factor
            avg_price = historical_data['close'].mean()
            price_scale = math.log10(max(avg_price, 1))
            
            # Calculate volatility-based scaling
            returns = np.log(historical_data['close'] / historical_data['close'].shift(1))
            volatility = returns.std() * math.sqrt(len(historical_data))
            
            # Calculate volume profile
            avg_volume = historical_data['volume'].mean()
            volume_std = historical_data['volume'].std()
            
            return {
                'macd_threshold': max(0.00005, volatility * price_scale * 0.05),
                'bb_threshold': volatility * avg_price * 0.005,
                'rsi_overbought': min(80 + volatility * 10, 85),
                'rsi_oversold': max(20 - volatility * 10, 15),
                'adx_threshold': 15 + volatility * 5,
                'volume_threshold': avg_volume + volume_std * 0.3
            }
        except Exception as e:
            logger.error(f"Error calculating dynamic thresholds for {symbol}: {e}")
            return None

    async def get_trend(self, symbol: str) -> str:
        historical_data = await self.market_data.get_historical_data(symbol, self.timeframe, limit=100)  # Reduced from 10000
        close_prices = historical_data['close']
        
        # Use even shorter EMA periods for faster response
        ema_short = ta.ema(close_prices, length=3)  # Reduced from 5
        ema_medium = ta.ema(close_prices, length=7)  # Reduced from 10
        ema_long = ta.ema(close_prices, length=14)  # Reduced from 20
        
        # Check multiple EMA crossovers for trend confirmation
        short_above_medium = ema_short.iloc[-1] > ema_medium.iloc[-1]
        short_above_long = ema_short.iloc[-1] > ema_long.iloc[-1]
        medium_above_long = ema_medium.iloc[-1] > ema_long.iloc[-1]
        
        # Calculate price momentum over a shorter period
        momentum = (close_prices.iloc[-1] - close_prices.iloc[-3]) / close_prices.iloc[-3]  # Changed from -5 to -3
        
        # Add RSI for additional trend confirmation
        rsi = ta.rsi(close_prices, length=7)  # 7-period RSI
        
        # Adjust trend determination logic
        if short_above_medium and short_above_long and medium_above_long and momentum > 0 and rsi.iloc[-1] > 60:
            return 'strong_up'
        elif short_above_medium and short_above_long and momentum > 0 and rsi.iloc[-1] > 50:
            return 'up'
        elif not short_above_medium and not short_above_long and not medium_above_long and momentum < 0 and rsi.iloc[-1] < 40:
            return 'strong_down'
        elif not short_above_medium and not short_above_long and momentum < 0 and rsi.iloc[-1] < 50:
            return 'down'
        else:
            return 'neutral'

    # Add a method to log the trend
    async def log_trend(self, symbol: str):
        trend = await self.get_trend(symbol)
        logger.info(f"Current trend for {symbol}: {trend}")

    async def get_cached_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        cache_key = f"{symbol}_{interval}_{limit}"
        if cache_key in self.historical_data_cache:
            return self.historical_data_cache[cache_key]
        
        # Fetch more data than required to ensure enough for feature preparation
        data = await self.market_data.get_historical_data(symbol, interval, limit=max(limit, 10000))
        if data is not None:
            self.historical_data_cache[cache_key] = data
        return data

    def is_cool_down_active(self, symbol: str) -> bool:
        last_attempt = self.last_trade_attempt.get(symbol, 0)
        return time.time() - last_attempt < self.cool_down_period

    def prepare_training_data(self, historical_data):
        all_X = []
        all_y = []
        for symbol, data in historical_data.items():
            X = self.prepare_features(data)
            y = data['close'].shift(-1).dropna().values
            X = X[:-1]  # Remove the last row to match y's length
            all_X.append(X)
            all_y.append(y)
        return np.vstack(all_X), np.concatenate(all_y)
    
    async def train_model(self):
        current_time = time.time()
        if current_time - self.last_model_training < self.training_interval:
            return

        try:
            logger.info("Training price prediction model...")
            historical_data = {}
            for symbol in self.symbols:
                data = await self.market_data.get_historical_data(symbol, self.timeframe, limit=10000)
                if data is not None:
                    historical_data[symbol] = data

            if historical_data:
                X, y = self.prepare_training_data(historical_data)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Check for infinite or very large values
                if np.isinf(X_train).any() or np.isinf(y_train).any():
                    logger.warning("Infinite values found in training data. Removing them.")
                    mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
                    X_train = X_train[mask]
                    y_train = y_train[mask]
                
                self.price_predictor.train(X_train, y_train)
                self.last_model_training = current_time
                
                # Evaluate the model
                y_pred = self.price_predictor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"Model evaluation - MSE: {mse:.4f}, R2: {r2:.4f}")
            else:
                logger.warning("No historical data available for model training")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}", exc_info=True)
            logger.warning("Failed to train the model. Using the previous model if available.")

    async def collect_historical_data(self) -> Dict[str, pd.DataFrame]:
        historical_data = {}
        min_length = float('inf')
        for symbol in self.symbols:
            data = await self.market_data.get_historical_data(symbol, '1m', limit=10000)
            if data is not None:
                historical_data[symbol] = data
                min_length = min(min_length, len(data))
        
        # Ensure all DataFrames have the same length
        for symbol, data in historical_data.items():
            historical_data[symbol] = data.tail(min_length)
        
        return historical_data

    async def execute_trade(self, symbol: str, signal: str) -> Optional[Trade]:
        try:
            account_balance = self.exchange_client.get_account_balance()
            current_price = float((self.exchange_client.get_symbol_ticker(symbol))['price'])
            volatility = await self.calculate_volatility(symbol)
            
            # Get symbol information
            symbol_info = self.exchange_client.get_symbol_info(symbol)
            lot_size_filter = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
            min_notional_filter = next(filter(lambda x: x['filterType'] == 'MIN_NOTIONAL', symbol_info['filters']))
            
            # Extract lot size constraints
            step_size = float(lot_size_filter['stepSize'])
            min_qty = float(lot_size_filter['minQty'])
            min_notional = float(min_notional_filter['minNotional'])
            
            # Calculate position size based on risk management
            position_size = await self.calculate_position_size(symbol, account_balance, volatility, current_price)
            
            # Validate against minimum notional value
            if position_size * current_price < min_notional:
                logger.warning(f"Position size too small for {symbol}. Minimum notional value: {min_notional} USDT")
                return None
            
            # Round down to the nearest valid quantity
            precision = int(round(-math.log10(float(step_size))))
            quantity = math.floor(position_size * (10 ** precision)) / (10 ** precision)
            
            # Validate minimum quantity
            if quantity < min_qty:
                logger.warning(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
                return None
                
            # Execute the trade
            if signal == 'buy':
                order = self.exchange_client.create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=quantity
                )
            elif signal == 'sell':
                order = self.exchange_client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity
                )
            else:
                logger.warning(f"Invalid signal {signal} for {symbol}")
                return None

            if order['status'] == 'FILLED':
                trade = Trade(
                    symbol=symbol,
                    entry_price=float(order['fills'][0]['price']),
                    amount=float(order['executedQty']),
                    side=signal,
                    entry_time=datetime.now(),
                    status=TradeStatus.OPEN
                )
                logger.info(f"Executed {signal} trade for {symbol}: {trade}")
                return trade
            else:
                logger.warning(f"Order not filled for {symbol}: {order}")
                return None

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return None
    
    async def get_lot_size(self, symbol: str) -> float:
        try:
            # Fetch symbol information from the exchange
            symbol_info = self.exchange_client.get_symbol_info(symbol)
            
            # Extract the LOT_SIZE filter
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                return float(lot_size_filter['stepSize'])
            else:
                logger.warning(f"LOT_SIZE filter not found for {symbol}. Using default value of 0.00001")
                return 0.00001
        except Exception as e:
            logger.error(f"Error fetching lot size for {symbol}: {e}")
            return 0.00001  # Return a default value

    async def calculate_position_size(self, symbol: str, account_balance: float, volatility: float, current_price: float) -> float:
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [account_balance, volatility, current_price]):
                logger.error(f"Invalid input types for position size calculation: balance={type(account_balance)}, volatility={type(volatility)}, price={type(current_price)}")
                return 0.0
                
            if account_balance <= 0 or current_price <= 0:
                logger.error(f"Invalid values for position size calculation: balance={account_balance}, price={current_price}")
                return 0.0

            # Get symbol info for validation
            symbol_info = self.exchange_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.warning(f"Could not fetch symbol info for {symbol}")
                return 0.0

            # Extract filters
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
            
            if not all([lot_size_filter, min_notional_filter]):
                logger.warning(f"Required filters not found for {symbol}")
                return 0.0

            min_qty = float(lot_size_filter['minQty'])
            step_size = float(lot_size_filter['stepSize'])
            min_notional = float(min_notional_filter['minNotional'])

            # Calculate base position size (more aggressive)
            risk_amount = account_balance * self.risk_per_trade * 1.5  # Increased risk amount
            
            # Calculate dynamic stop loss distance (tighter)
            atr = await self.calculate_atr(symbol, period=14, smoothing=2)
            stop_loss_distance = max(atr * 1.2, current_price * 0.004)  # Reduced from 1.5 ATR and 0.005
            
            # Calculate position size
            position_size = risk_amount / max(stop_loss_distance, 0.00001)
            
            # Apply volatility adjustment (less conservative)
            volatility_factor = 1 / (1 + max(volatility, 0.00001))
            position_size *= volatility_factor * 1.2  # Increased by 20%
            
            # Consider market liquidity
            avg_volume = await self.get_average_volume(symbol, period=30)  # Reduced from 50
            max_trade_volume = avg_volume * 0.003  # Increased from 0.002
            position_size = min(position_size, max_trade_volume)
            
            # Apply maximum position size limit
            max_position_value = account_balance * self.max_position_size * 1.2  # Increased by 20%
            position_size = min(position_size, max_position_value / max(current_price, 0.00001))
            
            # Round to valid step size
            precision = int(round(-math.log10(float(step_size))))
            position_size = math.floor(position_size * (10 ** precision)) / (10 ** precision)
            
            # Ensure minimum notional value
            if position_size * current_price < min_notional:
                position_size = math.ceil(min_notional / current_price / step_size) * step_size
            
            # Final validation with more lenient minimums
            if position_size < min_qty * 0.8:  # 20% below minimum quantity
                logger.warning(f"Position size {position_size} below adjusted minimum quantity {min_qty * 0.8} for {symbol}")
                return 0.0
                
            if position_size * current_price < min_notional * 0.8:  # 20% below minimum notional
                logger.warning(f"Position value {position_size * current_price:.8f} below adjusted minimum notional {min_notional * 0.8} for {symbol}")
                return 0.0
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    async def update_open_trades(self):
        for symbol, trade in list(self.open_trades.items()):
            current_price = await self.market_data.get_current_price(symbol)
            if current_price is None:
                continue

            if trade.side == 'BUY':
                if current_price > trade.entry_price:
                    new_stop_loss = max(trade.stop_loss, current_price * 0.995)  # 0.5% trailing stop-loss
                    if new_stop_loss > trade.stop_loss:
                        trade.stop_loss = new_stop_loss
                elif current_price <= trade.stop_loss:
                    await self.close_trade(symbol, trade, current_price, 'Stop-loss hit')
            else:  # SELL
                if current_price < trade.entry_price:
                    new_stop_loss = min(trade.stop_loss, current_price * 1.005)  # 0.5% trailing stop-loss
                    if new_stop_loss < trade.stop_loss:
                        trade.stop_loss = new_stop_loss
                elif current_price >= trade.stop_loss:
                    await self.close_trade(symbol, trade, current_price, 'Stop-loss hit')
        
    async def manage_open_trades(self):
        for symbol, trade in list(self.open_trades.items()):
            current_price = await self.market_data.get_latest_price(symbol)
            if current_price is None:
                logger.error(f"Unable to get current price for {symbol}")
                continue

            trade_duration = time.time() - trade.entry_time
            pnl = self.calculate_pnl(trade, current_price)

            # Update trailing stop loss
            if trade.side == 'buy':
                new_stop_loss = max(trade.stop_loss, current_price * (1 - self.trailing_stop_percentage))
            else:
                new_stop_loss = min(trade.stop_loss, current_price * (1 + self.trailing_stop_percentage))
            trade.stop_loss = new_stop_loss

            if trade_duration > self.max_trade_duration:
                await self.close_trade(trade, current_price, "Max duration reached")
            elif (trade.side == 'buy' and current_price <= trade.stop_loss) or \
                 (trade.side == 'sell' and current_price >= trade.stop_loss):
                await self.close_trade(trade, current_price, "Stop loss hit")
            elif (trade.side == 'buy' and current_price >= trade.take_profit) or \
                 (trade.side == 'sell' and current_price <= trade.take_profit):
                await self.close_trade(trade, current_price, "Take profit reached")
            elif self.should_close_trade(trade, current_price):
                await self.close_trade(trade, current_price, "Strategy condition met")

    async def close_trade(self, trade: Trade, current_price: float):
        close_side = "SELL" if trade.side == "buy" else "BUY"
        try:
            order = self.exchange_client.create_order(
                symbol=trade.symbol,
                side=close_side,
                type="MARKET",
                quantity=trade.amount
            )

            if order['status'] == 'FILLED':
                trade.exit_price = float(order['fills'][0]['price'])
                trade.status = TradeStatus.CLOSED
                trade.pnl = self.calculate_pnl(trade, trade.exit_price)
                logger.info(f"Closed trade: {trade}")
                del self.open_trades[trade.symbol]
                self.trade_history.append(trade)
                self.update_strategy_parameters(trade)
            else:
                logger.warning(f"Order not filled immediately: {order}")
        except Exception as e:
            logger.error(f"Error closing trade {trade.symbol}: {e}")

    def update_strategy_parameters(self, trade: Trade):
        if trade.pnl > 0:
            self.trailing_stop_percentage *= 1.01
            self.risk_per_trade *= 1.01
        else:
            self.trailing_stop_percentage *= 0.99
            self.risk_per_trade *= 0.99

        self.trailing_stop_percentage = min(max(self.trailing_stop_percentage, 0.005), 0.03)
        self.risk_per_trade = min(max(self.risk_per_trade, 0.005), 0.02)

    async def should_close_trade(self, trade: Trade, current_price: float) -> bool:
        try:
            # Get recent price action
            historical_data = await self.market_data.get_historical_data(trade.symbol, self.timeframe, limit=30)
            if historical_data is None:
                return False

            close_prices = historical_data['close']
            high_prices = historical_data['high']
            low_prices = historical_data['low']

            # Calculate short-term indicators
            ema3 = ta.ema(close_prices, length=3)  # Reduced from 5
            ema7 = ta.ema(close_prices, length=7)  # Reduced from 10
            rsi = ta.rsi(close_prices, length=7)
            
            # Calculate price momentum
            momentum = (close_prices.iloc[-1] - close_prices.iloc[-3]) / close_prices.iloc[-3]

            if trade.side == "buy":
                # Close long positions if trend weakens
                if (ema3.iloc[-1] < ema7.iloc[-1] and 
                    momentum < -0.001 and  # Small negative momentum
                    rsi.iloc[-1] < 45):    # RSI showing weakness
                    return True
                
                # Take profit at 0.3% gain
                if current_price >= trade.entry_price * 1.003:
                    return True
                    
            else:  # sell
                # Close short positions if trend weakens
                if (ema3.iloc[-1] > ema7.iloc[-1] and 
                    momentum > 0.001 and   # Small positive momentum
                    rsi.iloc[-1] > 55):    # RSI showing strength
                    return True
                
                # Take profit at 0.3% gain
                if current_price <= trade.entry_price * 0.997:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error in should_close_trade: {e}")
            return False

    async def get_account_balance(self):
        return self.exchange_client.get_account_balance()

    async def convert_all_assets_to_usdt(self):
        logger.info("Converting all assets to USDT...")
        try:
            total_converted = self.exchange_client.convert_all_assets_to_usdt()
            logger.info(f"Successfully converted assets to USDT. Total converted: {total_converted}")
            
            # Update the account balance after conversion
            new_balance = await self.get_account_balance()
            logger.info(f"Updated USDT balance after conversion: {new_balance}")
            
            return total_converted
        except Exception as e:
            logger.error(f"Error converting assets to USDT: {e}")
            return 0.0

    def get_open_trades_count(self) -> int:
        return len(self.open_trades)

    async def get_trade_history(self) -> List[Trade]:
        return self.trade_history

    async def calculate_performance(self) -> Dict[str, float]:
        total_profit = sum(trade.pnl for trade in self.trade_history)
        win_rate = sum(1 for trade in self.trade_history if trade.pnl > 0) / len(self.trade_history) if self.trade_history else 0
        return {
            "total_profit": total_profit,
            "win_rate": win_rate,
            "open_trades": len(self.open_trades),
            "closed_trades": len(self.trade_history)
        }

    async def close_all_trades(self):
        if not self.open_trades:
            logger.info("No open trades to close")
            return

        for symbol, trade in list(self.open_trades.items()):
            current_price = await self.market_data.get_latest_price(symbol)
            if current_price is None:
                logger.error(f"Unable to get current price for {symbol}")
                continue
            await self.close_trade(trade, current_price)
        await self.convert_all_assets_to_usdt()
        self.open_trades.clear()

    def log_open_trades(self):
        if not self.open_trades:
            logger.info("No open trades")
        else:
            logger.info("Current open trades:")
            for symbol, trade in self.open_trades.items():
                logger.info(f"  {symbol}: {trade}")

    async def check_and_close_trades(self):
        for symbol, trade in list(self.open_trades.items()):
            current_price = await self.market_data.get_latest_price(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                continue

            pair_config = self.risk_manager.get_pair_config(symbol)
            profit_percentage = pair_config['profit_percentage'] / 100
            stop_loss_percentage = pair_config['stop_loss_percentage'] / 100

            if trade.side == 'buy':
                profit_price = trade.entry_price * (1 + profit_percentage)
                stop_loss_price = trade.entry_price * (1 - stop_loss_percentage)
                if current_price >= profit_price:
                    logger.info(f"Closing trade {trade.trade_id} for profit. Current price: {current_price}, Profit price: {profit_price}")
                    await self.close_trade(trade, current_price)
                elif current_price <= stop_loss_price:
                    logger.info(f"Closing trade {trade.trade_id} for stop loss. Current price: {current_price}, Stop loss price: {stop_loss_price}")
                    await self.close_trade(trade, current_price)
            else:  # sell
                profit_price = trade.entry_price * (1 - profit_percentage)
                stop_loss_price = trade.entry_price * (1 + stop_loss_percentage)
                if current_price <= profit_price:
                    logger.info(f"Closing trade {trade.trade_id} for profit. Current price: {current_price}, Profit price: {profit_price}")
                    await self.close_trade(trade, current_price)
                elif current_price >= stop_loss_price:
                    logger.info(f"Closing trade {trade.trade_id} for stop loss. Current price: {current_price}, Stop loss price: {stop_loss_price}")
                    await self.close_trade(trade, current_price)

    def log_trade_attempt(self, symbol: str, side: str, quantity: float, price: float):
        logger.info(f"Attempting to execute {side} trade for {symbol}")
        logger.info(f"Trade details - Symbol: {symbol}, Side: {side}, Quantity: {quantity}, Price: {price}, Notional: {quantity * price}")

    def adapt_parameters(self):
        # Adjust strategy parameters based on recent performance
        recent_trades = self.get_recent_trades(50)
        win_rate = self.calculate_win_rate(recent_trades)
        
        if win_rate < 0.4:
            self.bb_period = max(10, self.bb_period - 1)
            self.stop_loss_percentage *= 1.1
            self.take_profit_percentage *= 0.9
        elif win_rate > 0.6:
            self.bb_period = min(30, self.bb_period + 1)
            self.stop_loss_percentage *= 0.9
            self.take_profit_percentage *= 1.1

    def get_recent_trades(self, n: int) -> List[Trade]:
        return self.trade_history[-n:]

    def calculate_win_rate(self, trades: List[Trade]) -> float:
        if not trades:
            return 0
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        return winning_trades / len(trades)

    async def rebalance_account(self):
        logger.info("Rebalancing account...")
        account_info = self.exchange_client.get_account()
        for balance in account_info['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            if asset != 'USDT' and free_balance > 0:
                symbol = f"{asset}USDT"
                try:
                    current_price = await self.market_data.get_latest_price(symbol)
                    if current_price is None:
                        continue
                    notional = free_balance * current_price
                    if notional > 10:  # Only sell if the notional value is greater than 10 USDT
                        order = self.exchange_client.create_order(
                            symbol=symbol,
                            side="SELL",
                            type="MARKET",
                            quantity=free_balance
                        )
                        logger.info(f"Rebalanced {asset}: {order}")
                    await self.convert_all_assets_to_usdt()
                except Exception as e:
                    logger.error(f"Error rebalancing {asset}: {e}")
        logger.info("Account rebalancing completed")

    async def get_asset_balance(self, asset: str) -> float:
        account_info = self.exchange_client.get_account()
        for balance in account_info['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0

    def calculate_pnl(self, trade: Trade, current_price: float) -> float:
        if trade.side == 'buy':
            pnl = (current_price - trade.entry_price) * trade.amount
        else:  # sell
            pnl = (trade.entry_price - current_price) * trade.amount
        return pnl

    async def calculate_volatility(self, symbol: str) -> float:
        # Increase the limit to get more data points for a smoother calculation
        historical_data = await self.market_data.get_historical_data(symbol, '1m', limit=10000)
        
        if historical_data.empty or len(historical_data) < 2:
            return 0.005  # Lower default volatility for 1-minute chart
        
        # Calculate returns
        returns = np.log(historical_data['close'] / historical_data['close'].shift(1))
        
        # Use exponentially weighted standard deviation for more emphasis on recent volatility
        volatility = returns.ewm(span=20, adjust=False).std().iloc[-1]
        
        # Annualize the volatility
        annualized_volatility = volatility * np.sqrt(525600)  # sqrt(minutes in a year)
        
        # Cap the volatility to avoid extreme values
        max_volatility = 0.1  # 10% max volatility
        capped_volatility = min(annualized_volatility, max_volatility)
        
        return capped_volatility

    async def calculate_atr(self, symbol: str, timeframe: str = '1m') -> float:
        """Calculate Average True Range"""
        try:
            klines = await self.exchange.get_klines(symbol, timeframe, limit=self.atr_period + 1)
            high = pd.Series([float(k[2]) for k in klines])
            low = pd.Series([float(k[3]) for k in klines])
            close = pd.Series([float(k[4]) for k in klines])
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
            return float(atr)
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {str(e)}")
            return 0.0

    async def calculate_position_size(self, symbol: str, signal_type: str, signal_strength: float) -> float:
        """Calculate position size based on ATR and account balance"""
        try:
            balance = await self.exchange.get_balance()
            current_price = await self.exchange.get_price(symbol)
            atr = await self.calculate_atr(symbol)
            
            if atr == 0 or current_price == 0:
                return 0.0
                
            # Risk per trade based on ATR and signal strength
            risk_per_trade = min(balance * 0.01 * signal_strength, self.max_position_size_usd)  # Max 1% risk per trade
            
            # Position size based on ATR
            position_size = risk_per_trade / (atr * self.position_size_atr_multiplier)
            
            # Convert to units of the asset
            position_size_units = position_size / current_price
            
            # Apply minimum position size
            if position_size < self.min_position_size_usd:
                return 0.0
                
            # Apply maximum position size
            position_size_units = min(position_size_units, self.max_position_size_usd / current_price)
            
            return position_size_units
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 0.0

    async def get_average_volume(self, symbol: str, period: int = 50) -> float:
        """Calculate average volume over a period."""
        try:
            historical_data = await self.market_data.get_historical_data(symbol, self.timeframe, limit=period)
            if historical_data is None or len(historical_data) < period:
                return 0.0
            return float(historical_data['volume'].mean())
        except Exception as e:
            logger.error(f"Error calculating average volume for {symbol}: {e}")
            return 0.0

    def prepare_features(self, df: pd.DataFrame) -> np.array:
        df = df.copy()
        
        # Adjust moving averages for shorter timeframes
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
        
        # RSI with shorter period
        df['RSI'] = ta.rsi(df['close'], length=7)
        
        # MACD with faster settings
        macd = ta.macd(df['close'], fast=6, slow=13, signal=4)
        df['MACD'] = macd['MACD_6_13_4']
        df['MACD_Signal'] = macd['MACDs_6_13_4']
        
        # ATR with shorter period
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=7)
        
        # Bollinger Bands with shorter period
        bbands = ta.bbands(df['close'], length=10)
        df['Bollinger_Upper'] = bbands['BBU_10_2.0']
        df['Bollinger_Middle'] = bbands['BBM_10_2.0']
        df['Bollinger_Lower'] = bbands['BBL_10_2.0']
        
        # ADX with shorter period
        adx = ta.adx(df['high'], df['low'], df['close'], length=7)
        df['ADX'] = adx['ADX_7']
        
        # OBV (no change needed)
        df['OBV'] = ta.obv(df['close'], df['volume'])
        
        # Add price and volume ratios
        df['price_to_sma_ratio'] = df['close'] / df['SMA_5']
        df['volume_to_avg_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
        
        # Add more advanced features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=10).std() * np.sqrt(1440)  # Annualize for 1-minute data
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Momentum (shorter lookback)
        df['momentum'] = df['close'] - df['close'].shift(2)
        
        # Add some new features for short-term trading
        df['price_acceleration'] = df['price_change'] - df['price_change'].shift(1)
        df['volume_acceleration'] = df['volume_change'] - df['volume_change'].shift(1)
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        features = df[['open', 'high', 'low', 'close', 'volume', 'SMA_5', 'SMA_10', 'EMA_5', 'RSI', 'MACD', 'ATR',
                       'Bollinger_Upper', 'Bollinger_Lower', 'ADX', 'OBV', 'price_to_sma_ratio', 'volume_to_avg_ratio',
                       'price_change', 'volume_change', 'momentum', 'volatility', 'log_return',
                       'price_acceleration', 'volume_acceleration', 'high_low_range', 'close_to_high']].values
        
        # Remove infinite or very large values
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        
        return features.astype(np.float32)

    def simplify_trade(self, trade: Trade) -> dict:
        """Convert a Trade object to a simple dictionary."""
        return {
            "trade_id": str(trade.trade_id),
            "symbol": trade.symbol,
            "side": trade.side,
            "amount": float(trade.amount),
            "entry_price": float(trade.entry_price),
            "exit_price": float(trade.exit_price) if trade.exit_price else None,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
            "status": trade.status.value,
            "pnl": float(trade.pnl),
        }

    @staticmethod
    def calculate_rsi(prices, period=9, smoothing=2):
        if len(prices) < period + smoothing:
            return None  # Not enough data

        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        
        if down == 0:
            rs = float('inf')
        else:
            rs = up/down
        
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(period - 1) + upval)/period
            down = (down*(period - 1) + downval)/period
            
            if down == 0:
                rs = float('inf')
            else:
                rs = up/down
            
            rsi[i] = 100. - 100./(1. + rs)

        # Apply smoothing
        smoothed_rsi = np.convolve(rsi, np.ones(smoothing), 'valid') / smoothing

        return smoothed_rsi[-1]

    @staticmethod
    def adaptive_rsi(prices, short_period=5, long_period=14, smoothing=2):
        if len(prices) < max(short_period, long_period) + smoothing:
            return None  # Not enough data

        short_rsi = ScalpingStrategy.calculate_rsi(prices, period=short_period, smoothing=smoothing)
        long_rsi = ScalpingStrategy.calculate_rsi(prices, period=long_period, smoothing=smoothing)
        
        if short_rsi is None or long_rsi is None:
            return None

        return (short_rsi + long_rsi) / 2

    @staticmethod
    def dynamic_rsi(prices, base_period=9, volatility_window=30, smoothing=2):
        if len(prices) < base_period + volatility_window + smoothing:
            return None  # Not enough data

        volatility = np.std(prices[-volatility_window:])
        dynamic_period = int(base_period * (1 + volatility))
        
        return ScalpingStrategy.calculate_rsi(prices, period=dynamic_period, smoothing=smoothing)

    def prepare_training_data(self, historical_data: Dict[str, pd.DataFrame]) -> Tuple[np.array, np.array]:
        all_X = []
        all_y = []
        for symbol, data in historical_data.items():
            X = self.prepare_features(data)
            y = data['close'].shift(-1).dropna().values
            
            # Ensure X and y have the same number of samples
            min_samples = min(len(X), len(y))
            X = X[:min_samples]
            y = y[:min_samples]
            
            all_X.append(X)
            all_y.append(y)
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        # Ensure X and y have the same number of samples
        min_samples = min(len(X), len(y))
        X = X[:min_samples]
        y = y[:min_samples]
        
        logger.info(f"Prepared training data. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    async def should_trade(self, symbol: str) -> bool:
        if symbol not in self.last_trade_attempt:
            return True
        time_since_last_attempt = time.time() - self.last_trade_attempt[symbol]
        volatility = await self.calculate_volatility(symbol)
        dynamic_cool_down = max(300, min(3600, int(3600 * volatility)))
        return time_since_last_attempt >= dynamic_cool_down

# Usage example
"""
async def main():
    client = BinanceClientWrapper(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET)
    market_data = MarketData(client)
    strategy = ScalpingStrategy(market_data, client)
    
    # Run the strategy for a short time as an example
    strategy_task = asyncio.create_task(strategy.run())
    await asyncio.sleep(300)  # Run for 5 minutes
    strategy_task.cancel()
    
    # Print performance
    performance = await strategy.calculate_performance()
    print(f"Strategy performance: {performance}")

if __name__ == "__main__":
    asyncio.run(main())
"""