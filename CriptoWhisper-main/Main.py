import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import ccxt.async_support as ccxt_async
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
# ✅ HFT: Технические индикаторы отключены для HFT (позиции 1-180 сек)
# from ta import trend, volatility
# from ta.momentum import RSIIndicator, StochasticOscillator, UltimateOscillator
# from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
# from ta.volatility import AverageTrueRange
# from ta.trend import IchimokuIndicator, PSARIndicator, CCIIndicator, TRIXIndicator, MACD
from dotenv import load_dotenv
from collections import deque
import signal
import copy
import json
import optuna
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

# Add new imports for order book and trade data processing
from collections import defaultdict

load_dotenv()

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

API_KEY = os.getenv("API_KEY", "TAHqn***xGhVmof2E")
API_SECRET = os.getenv("API_SECRET", "0QVQhcRx6SP1uZ****BJDIiQKqN1")

exchange_config = {
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
        'defaultSubType': 'linear',
        'adjustForTimeDifference': True,
        'recvWindow': 60000,  # Increased recvWindow to 60 seconds
        'demo': True
    },
    'timeout': 30000
}

class ProgressCallback(BaseCallback):
    """
    Callback for monitoring training progress
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_reported_progress = -1

    def _on_step(self) -> bool:
        progress = int((self.num_timesteps / self.total_timesteps) * 100)
        if progress > self.last_reported_progress and progress % 10 == 0:
            logging.info(f"Прогресс обучения: {progress}%")
            self.last_reported_progress = progress
        return True

class AdaptiveRiskManager:
    """
    Адаптивное управление рисками на основе Kelly Criterion для HFT
    Оптимизировано для краткосрочных позиций (1-180 сек)
    """
    def __init__(self, max_risk=0.02, min_risk=0.005):
        """
        Args:
            max_risk: Максимальный риск на сделку (2% для HFT)
            min_risk: Минимальный риск на сделку (0.5%)
        """
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.trade_history = deque(maxlen=100)  # Последние 100 сделок

    def calculate_position_size(self, balance, current_price, atr):
        """
        Расчет оптимального размера позиции на основе Half Kelly Criterion

        Returns:
            tuple: (amount, risk_fraction)
        """
        if len(self.trade_history) < 20:
            # Консервативный подход до накопления статистики
            kelly_fraction = self.min_risk
            logging.debug(f"Kelly: Недостаточно истории ({len(self.trade_history)}/20), используем min_risk={self.min_risk:.2%}")
        else:
            # Рассчитываем Kelly Criterion
            profits = [t['profit'] for t in self.trade_history]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]

            if len(winning_trades) == 0 or len(losing_trades) == 0:
                kelly_fraction = self.min_risk
                logging.debug("Kelly: Нет winning или losing trades, используем min_risk")
            else:
                win_rate = len(winning_trades) / len(self.trade_history)
                avg_win = np.mean(winning_trades)
                avg_loss = abs(np.mean(losing_trades))

                # Kelly formula: f = (bp - q) / b
                # где b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
                if avg_loss > 0:
                    b = avg_win / avg_loss  # Win/Loss ratio
                    kelly_fraction = (b * win_rate - (1 - win_rate)) / b
                else:
                    kelly_fraction = self.min_risk

                # Half Kelly для консерватизма (снижение волатильности)
                kelly_fraction = kelly_fraction * 0.5

                # Ограничиваем риск в допустимом диапазоне
                kelly_fraction = np.clip(kelly_fraction, self.min_risk, self.max_risk)

                logging.debug(
                    f"Kelly: win_rate={win_rate:.2%}, "
                    f"avg_win={avg_win:.6f}, avg_loss={avg_loss:.6f}, "
                    f"b={b:.2f}, kelly={kelly_fraction:.2%}"
                )

        # Дополнительная коррекция на волатильность рынка (важно для HFT)
        volatility_factor = min(atr / current_price, 0.05)  # Макс 5% волатильность
        # Снижаем риск при высокой волатильности
        risk_adjusted_fraction = kelly_fraction * (1 - volatility_factor * 10)
        risk_adjusted_fraction = max(risk_adjusted_fraction, self.min_risk)

        # Финальный размер позиции
        position_size = balance * risk_adjusted_fraction
        amount = position_size / current_price

        logging.info(
            f"💰 Position sizing: balance={balance:.2f}, "
            f"risk={risk_adjusted_fraction:.2%}, "
            f"position_size={position_size:.2f}, amount={amount:.6f}"
        )

        return amount, risk_adjusted_fraction

    def add_trade(self, profit, duration, win):
        """Добавление сделки в историю для расчета Kelly"""
        self.trade_history.append({
            'profit': profit,
            'duration': duration,
            'win': win,
            'timestamp': pd.Timestamp.now()
        })

    def get_statistics(self):
        """Получение статистики по сделкам"""
        if len(self.trade_history) == 0:
            return {}

        profits = [t['profit'] for t in self.trade_history]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        return {
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': (
                sum(winning_trades) / abs(sum(losing_trades))
                if losing_trades else float('inf')
            ),
            'total_profit': sum(profits)
        }

class TradingEnvironment(gym.Env):
    def __init__(self, data, norm_params=None, initial_balance=10, risk_percentage=0.01, short_term_threshold=10, long_term_threshold=50, history_size=100, window_size=20, max_position_duration=180):
        super(TradingEnvironment, self).__init__()
        logging.debug("Initializing TradingEnvironment for HFT")
        self.timestamps = data['timestamp'].reset_index(drop=True)
        self.data = data.drop(columns=['timestamp']).reset_index(drop=True)
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage  # Will be adaptive with Kelly
        self.short_term_threshold = short_term_threshold
        self.long_term_threshold = long_term_threshold
        self.window_size = window_size

        # HFT-specific parameters
        self.max_position_duration = max_position_duration  # Max 180 seconds for HFT
        self.sl_multiplier = 1.5  # Tight SL for HFT (1.5x ATR)
        self.tp_multiplier = 2.5  # TP for HFT (2.5x ATR, R:R = 1:1.67)

        # ✅ Kelly Criterion Risk Manager для адаптивного sizing
        self.risk_manager = AdaptiveRiskManager(max_risk=0.02, min_risk=0.005)

        if norm_params is None:
            self.means = self.data.mean()
            self.stds = self.data.std().replace(0, 1e-8)
        else:
            self.means = pd.Series(norm_params['means'])
            self.stds = pd.Series(norm_params['stds'])
        self.normalized_data = (self.data - self.means) / self.stds
        low = self.normalized_data.min().values - 1
        high = self.normalized_data.max().values + 1
        num_features = self.data.shape[1]
        self.observation_space = spaces.Box(
            low=np.tile(low, self.window_size).astype(np.float32),
            high=np.tile(high, self.window_size).astype(np.float32),
            shape=(self.window_size * num_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Изменено с 4 на 3
        self.obs_window = deque(maxlen=self.window_size)
        self.history = deque(maxlen=history_size)
        self.reset()
        self.save_state()

    def reset(self, *, seed=None, options=None):
        logging.debug("Resetting environment")
        self.balance = self.initial_balance
        self.previous_balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.entry_step = 0
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        self.positions = []
        self.position_size = 0
        self.units = 0
        self.balance_history = [self.balance]
        self.history.clear()
        self.obs_window.clear()
        initial_window = self.normalized_data.iloc[self.current_step:self.current_step + self.window_size]
        for _, row in initial_window.iterrows():
            self.obs_window.append(row.values.astype(np.float32))
        self.current_step += self.window_size
        self.save_state()
        return self._get_observation(), {}

    def _get_observation(self):
        if len(self.obs_window) < self.window_size:
            padding = [np.zeros(self.normalized_data.shape[1], dtype=np.float32)] * (self.window_size - len(self.obs_window))
            window = list(padding) + list(self.obs_window)
        else:
            window = list(self.obs_window)
        obs = np.concatenate(window)
        return obs.astype(np.float32)

    def save_state(self):
        state = {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'entry_step': self.entry_step,
            'current_step': self.current_step,
            'done': self.done,
            'total_profit': self.total_profit,
            'positions': copy.deepcopy(self.positions),
            'position_size': self.position_size,
            'units': self.units,
            'balance_history': copy.deepcopy(self.balance_history),
            'obs_window': copy.deepcopy(self.obs_window),
            'previous_balance': self.previous_balance
        }
        self.history.append(state)

    def load_state(self, steps_back=2):
        if len(self.history) >= steps_back:
            state = self.history[-steps_back]
            self.balance = state['balance']
            self.position = state['position']
            self.entry_price = state['entry_price']
            self.entry_step = state['entry_step']
            self.current_step = state['current_step']
            self.done = state['done']
            self.total_profit = state['total_profit']
            self.positions = copy.deepcopy(state['positions'])
            self.position_size = state['position_size']
            self.units = state['units']
            self.balance_history = copy.deepcopy(state['balance_history'])
            self.obs_window = copy.deepcopy(state['obs_window'])
            self.previous_balance = state['previous_balance']
            logging.debug("State loaded successfully")
        else:
            logging.warning("Недостаточно истории для отката")

    def detect_error(self):
        if self.balance < self.initial_balance * 0.5:
            logging.error("Баланс упал ниже половины начального значения")
            return True
        return False

    def handle_error(self):
        logging.info("Обработка ошибки путем отката состояния")
        self.load_state(steps_back=2)

    def _check_hft_stop_loss_take_profit(self, price):
        """
        Проверка SL/TP для HFT стратегии
        Учитывает краткосрочность позиций (1-180 сек)
        """
        if self.position is None:
            return False, 0

        atr = self.data['atr'].iloc[self.current_step] if 'atr' in self.data.columns else (price * 0.001)

        # Адаптивный multiplier на основе spread (для HFT важнее spread чем волатильность)
        spread = self.data['spread'].iloc[self.current_step] if 'spread' in self.data.columns else atr * 0.5
        volatility_factor = min(max(spread / price, 0.0005), 0.01)  # 0.05% - 1%

        if self.position == 'long':
            # Tight SL для HFT: 1.5x ATR
            stop_loss_price = self.entry_price - (self.sl_multiplier * atr)
            # TP для HFT: 2.5x ATR (Risk/Reward = 1:1.67)
            take_profit_price = self.entry_price + (self.tp_multiplier * atr)

            if price <= stop_loss_price:
                logging.info(f"⛔ HFT STOP LOSS triggered at {price:.8f} (entry: {self.entry_price:.8f})")
                return True, -0.5  # Penalty за SL
            elif price >= take_profit_price:
                logging.info(f"✅ HFT TAKE PROFIT triggered at {price:.8f} (entry: {self.entry_price:.8f})")
                return True, 0.3   # Bonus за TP

        elif self.position == 'short':
            stop_loss_price = self.entry_price + (self.sl_multiplier * atr)
            take_profit_price = self.entry_price - (self.tp_multiplier * atr)

            if price >= stop_loss_price:
                logging.info(f"⛔ HFT STOP LOSS triggered at {price:.8f} (entry: {self.entry_price:.8f})")
                return True, -0.5
            elif price <= take_profit_price:
                logging.info(f"✅ HFT TAKE PROFIT triggered at {price:.8f} (entry: {self.entry_price:.8f})")
                return True, 0.3

        return False, 0

    def _check_max_position_duration(self):
        """
        Проверка максимального времени удержания позиции для HFT
        Макс 180 секунд (3 минуты)
        """
        if self.position is None:
            return False

        duration = self.current_step - self.entry_step

        # Force close если позиция держится слишком долго
        if duration >= self.max_position_duration:
            logging.warning(f"⏱️ HFT MAX DURATION reached: {duration} steps (max: {self.max_position_duration})")
            return True

        return False

    def _calculate_hft_reward(self, profit, duration):
        """
        Продвинутая reward function для HFT
        Фокус на краткосрочных прибыльных сделках
        """
        atr = self.data['atr'].iloc[self.current_step] if 'atr' in self.data.columns else 0.001

        # 1. Базовая награда (risk-adjusted)
        base_reward = profit / (atr + 1e-8)

        # 2. Bonus за быстрые прибыльные сделки (HFT premium)
        if profit > 0 and duration <= 60:  # < 1 minute
            speed_bonus = 0.3 * (1 - duration / 60)  # Чем быстрее - тем больше bonus
        else:
            speed_bonus = 0

        # 3. Penalty за долгие убыточные позиции
        if profit < 0:
            duration_penalty = -0.002 * duration
        else:
            duration_penalty = 0

        # 4. Sharpe-like component (если есть история)
        if len(self.balance_history) >= 20:
            returns = np.diff(self.balance_history[-20:])
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / (returns.std() + 1e-8)
                sharpe_reward = sharpe * 0.2
            else:
                sharpe_reward = 0
        else:
            sharpe_reward = 0

        # 5. Win rate component
        if len(self.positions) >= 10:
            recent_profits = [p.get('profit', 0) for p in self.positions[-10:]]
            win_rate = sum(1 for p in recent_profits if p > 0) / len(recent_profits)
            win_rate_bonus = (win_rate - 0.5) * 0.2
        else:
            win_rate_bonus = 0

        # Total reward для HFT
        total_reward = base_reward + speed_bonus + duration_penalty + sharpe_reward + win_rate_bonus

        return total_reward

    def step(self, action):
        self.save_state()
        reward = 0
        info = {}
        if self.current_step >= len(self.data):
            self.done = True
            profit = self.balance - self.previous_balance
            duration = self.current_step - self.entry_step if self.position else 0
            reward = self._calculate_hft_reward(profit, duration)
            logging.debug(f"Эпизод завершен. Прибыль: {profit}, Награда: {reward}")
            return self._get_observation(), reward, self.done, False, info

        price = self.data['close'].iloc[self.current_step]
        timestamp = self.timestamps[self.current_step]
        atr = self.data['atr'].iloc[self.current_step] if 'atr' in self.data.columns else (price * 0.001)
        logging.debug(f"Текущий шаг: {self.current_step}, Цена: {price}, Время: {timestamp}")

        # ✅ КРИТИЧНО: Проверка HFT SL/TP ПЕРЕД любыми действиями
        should_close_sltp, sltp_reward = self._check_hft_stop_loss_take_profit(price)
        if should_close_sltp:
            reward += self._close_position(price, timestamp)
            reward += sltp_reward

        # ✅ КРИТИЧНО: Проверка максимального времени позиции (180 сек для HFT)
        if self._check_max_position_duration():
            logging.warning("Force closing position due to max duration")
            reward += self._close_position(price, timestamp)
            reward -= 0.3  # Penalty за force close

        # Выполнение действий агента
        if action == 0:
            logging.debug("Действие: Удерживать позицию")
            pass
        elif action == 1:
            if self.position == 'short':
                logging.info("Действие: Переключение с шорт на лонг")
                reward += self._close_position(price, timestamp)
            if self.position != 'long':
                logging.info("Действие: Открыть длинную позицию")
                self._open_position('long', price, timestamp, atr)
        elif action == 2:
            if self.position == 'long':
                logging.info("Действие: Переключение с лонг на шорт")
                reward += self._close_position(price, timestamp)
            if self.position != 'short':
                logging.info("Действие: Открыть короткую позицию")
                self._open_position('short', price, timestamp, atr)

        # ✅ Используем улучшенную HFT reward function
        profit = self.balance - self.previous_balance
        duration = self.current_step - self.entry_step if self.position else 0
        reward += self._calculate_hft_reward(profit, duration)

        logging.debug(f"Прибыль: {profit}, Duration: {duration}, Награда: {reward}")
        self.previous_balance = self.balance
        obs = self.normalized_data.iloc[self.current_step]
        self.obs_window.append(obs.values.astype(np.float32))
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            self.done = True
            logging.debug("Достигнут конец данных")

        self.balance_history.append(self.balance)

        if self.detect_error():
            self.handle_error()
            reward -= 10
            self.done = False

        return self._get_observation(), reward, self.done, False, info

    def _open_position(self, position_type, price, timestamp, atr):
        self.position = position_type
        self.entry_price = price
        self.entry_step = self.current_step

        # ✅ Используем Kelly Criterion для адаптивного sizing
        units, risk_fraction = self.risk_manager.calculate_position_size(
            balance=self.balance,
            current_price=price,
            atr=atr
        )

        self.position_size = self.balance * risk_fraction
        self.units = units

        self.positions.append({
            'entry_time': timestamp,
            'entry_price': price,
            'entry_step': self.current_step,
            'atr': atr,
            'risk_fraction': risk_fraction  # Сохраняем для анализа
        })
        logging.info(f"📊 Позиция открыта: {position_type} по цене {price:.8f}, размер: {self.units:.6f}, риск: {risk_fraction:.2%}")

    def _close_position(self, price, timestamp):
        if self.entry_price == 0:
            logging.warning("Попытка закрыть позицию без входной цены")
            return 0

        fee_rate = 0.001
        slippage = 0.001
        duration = self.current_step - self.entry_step
        atr = self.data['atr'].iloc[self.entry_step] if 'atr' in self.data.columns else (price * 0.001)

        if self.position == 'long':
            effective_price = price * (1 - slippage)
            profit = (effective_price - self.entry_price) * self.units
        else:
            effective_price = price * (1 + slippage)
            profit = (self.entry_price - effective_price) * self.units

        fee = self.position_size * fee_rate * 2
        profit -= fee
        self.balance += profit
        self.total_profit += profit

        # ✅ Добавляем сделку в risk_manager для Kelly Criterion
        is_win = profit > 0
        self.risk_manager.add_trade(profit=profit, duration=duration, win=is_win)

        # Используем HFT reward function
        reward = self._calculate_hft_reward(profit, duration)

        self.positions[-1].update({
            'exit_time': timestamp,
            'exit_price': price,
            'duration': duration,
            'profit': profit,
            'atr': atr,
            'win': is_win
        })

        profit_emoji = "✅" if profit > 0 else "❌"
        logging.info(
            f"{profit_emoji} Позиция закрыта: {self.position} по цене {price:.8f}, "
            f"Прибыль: {profit:.6f}, Duration: {duration}s"
        )

        # Логируем статистику Kelly
        stats = self.risk_manager.get_statistics()
        if stats:
            logging.info(
                f"📈 Kelly Stats: WinRate={stats['win_rate']:.2%}, "
                f"Trades={stats['total_trades']}, PF={stats['profit_factor']:.2f}"
            )

        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.units = 0

        return reward

def calculate_rvi(df, window=10):
    close_open = df['close'] - df['open']
    high_low = df['high'] - df['low']
    rvi = close_open / high_low
    rvi = rvi.rolling(window=window).mean()
    logging.debug("RVI рассчитан")
    return rvi

def add_technical_indicators(df):
    """Removed technical indicators analysis as per requirements"""
    logging.debug("Технические индикаторы удалены")
    return df

async def get_full_data(exchange, symbol, timeframe='5m', since=None, limit=2000):  # Изменено с '1m' на '5m'
    all_ohlcv = []
    logging.info(f"Начало получения данных для символа {symbol}")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                logging.debug("Нет новых данных для загрузки")
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 5 * 60 * 1000  # Изменено с 60 секунд на 5 минут
            if last_timestamp >= exchange.milliseconds():
                logging.debug("Достигнута текущая временная метка")
                break
        except Exception as e:
            logging.error(f"Ошибка при получении данных: {e}")
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    logging.info(f"Получено {len(df)} записей данных для символа {symbol}")
    return df

async def get_order_book_data(exchange, symbol, limit=50):
    """Fetch order book data from exchange"""
    try:
        orderbook = await exchange.fetch_order_book(symbol, limit=limit)
        return orderbook
    except Exception as e:
        logging.error(f"Ошибка при получении данных стакана: {e}")
        return None

async def get_trade_data(exchange, symbol, limit=100):
    """Fetch recent trades data from exchange"""
    try:
        trades = await exchange.fetch_trades(symbol, limit=limit)
        return trades
    except Exception as e:
        logging.error(f"Ошибка при получении данных ленты сделок: {e}")
        return None

def calculate_order_book_features(orderbook):
    """Calculate features from order book data"""
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {}
    
    bids = orderbook['bids']
    asks = orderbook['asks']
    
    if not bids or not asks:
        return {}
    
    # Best bid and ask prices
    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    
    # Bid-ask spread
    spread = best_ask - best_bid
    relative_spread = spread / ((best_bid + best_ask) / 2) if best_bid + best_ask > 0 else 0
    
    # Market depth
    bid_depth = sum([bid[1] for bid in bids])
    ask_depth = sum([ask[1] for ask in asks])
    depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if bid_depth + ask_depth > 0 else 0
    
    # Large orders detection
    bid_volumes = [bid[1] for bid in bids]
    ask_volumes = [ask[1] for ask in asks]
    avg_bid_volume = np.mean(bid_volumes) if bid_volumes else 0
    avg_ask_volume = np.mean(ask_volumes) if ask_volumes else 0
    
    large_bids = sum(1 for vol in bid_volumes if vol > avg_bid_volume * 2)
    large_asks = sum(1 for vol in ask_volumes if vol > avg_ask_volume * 2)
    
    # Price levels
    price_levels = len(bids) + len(asks)
    
    # Additional features for signal generation
    # Weighted average prices
    weighted_bid_price = sum([bid[0] * bid[1] for bid in bids]) / bid_depth if bid_depth > 0 else 0
    weighted_ask_price = sum([ask[0] * ask[1] for ask in asks]) / ask_depth if ask_depth > 0 else 0
    
    # Mid-price
    mid_price = (best_bid + best_ask) / 2
    
    # Additional order book features for PnL maximization
    # Volume-weighted average prices
    vwap_bid = sum([bid[0] * bid[1] for bid in bids]) / bid_depth if bid_depth > 0 else 0
    vwap_ask = sum([ask[0] * ask[1] for ask in asks]) / ask_depth if ask_depth > 0 else 0
    
    # Order book slope (price depth)
    price_depth = 0
    if len(bids) > 1 and len(asks) > 1:
        bid_slope = (bids[0][0] - bids[-1][0]) / len(bids)
        ask_slope = (asks[-1][0] - asks[0][0]) / len(asks)
        price_depth = (bid_slope + ask_slope) / 2
    
    # Liquidity imbalance at different levels
    top_5_bid_volume = sum([bid[1] for bid in bids[:5]])
    top_5_ask_volume = sum([ask[1] for ask in asks[:5]])
    liquidity_imbalance = (top_5_bid_volume - top_5_ask_volume) / (top_5_bid_volume + top_5_ask_volume) if top_5_bid_volume + top_5_ask_volume > 0 else 0
    
    # Market impact estimation
    market_impact = (vwap_ask - vwap_bid) / mid_price if mid_price > 0 else 0
    
    # Order book thickness
    bid_thickness = len(bids) / 50  # Normalized by max limit
    ask_thickness = len(asks) / 50  # Normalized by max limit
    
    return {
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread': spread,
        'relative_spread': relative_spread,
        'bid_depth': bid_depth,
        'ask_depth': ask_depth,
        'depth_imbalance': depth_imbalance,
        'large_bids': large_bids,
        'large_asks': large_asks,
        'price_levels': price_levels,
        'weighted_bid_price': weighted_bid_price,
        'weighted_ask_price': weighted_ask_price,
        'mid_price': mid_price,
        'vwap_bid': vwap_bid,
        'vwap_ask': vwap_ask,
        'price_depth': price_depth,
        'liquidity_imbalance': liquidity_imbalance,
        'market_impact': market_impact,
        'bid_thickness': bid_thickness,
        'ask_thickness': ask_thickness
    }

def calculate_trade_features(trades):
    """Calculate features from trade data"""
    if not trades:
        return {}
    
    # Convert to DataFrame for easier processing
    df_trades = pd.DataFrame(trades)
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'], unit='ms')
    
    # Trade volume and frequency
    total_volume = df_trades['amount'].sum()
    trade_count = len(df_trades)
    
    # Buy/sell imbalance
    buy_volume = df_trades[df_trades['side'] == 'buy']['amount'].sum() if 'side' in df_trades.columns else 0
    sell_volume = df_trades[df_trades['side'] == 'sell']['amount'].sum() if 'side' in df_trades.columns else 0
    volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if buy_volume + sell_volume > 0 else 0
    
    # Price movement
    if len(df_trades) > 1:
        price_changes = df_trades['price'].diff().dropna()
        avg_price_change = price_changes.mean()
        price_volatility = price_changes.std()
    else:
        avg_price_change = 0
        price_volatility = 0
    
    # Trade size analysis
    avg_trade_size = df_trades['amount'].mean()
    large_trades = sum(1 for amount in df_trades['amount'] if amount > avg_trade_size * 2)
    
    # Time-based features
    if len(df_trades) > 1:
        time_diffs = df_trades['timestamp'].diff().dt.total_seconds().dropna()
        avg_time_between_trades = time_diffs.mean()
        trade_frequency = 1 / avg_time_between_trades if avg_time_between_trades > 0 else 0
    else:
        avg_time_between_trades = 0
        trade_frequency = 0
    
    # Additional trade features for PnL maximization
    # Trade size distribution
    if len(df_trades) > 0:
        trade_sizes = df_trades['amount'].values
        size_std = np.std(trade_sizes)
        size_skewness = np.mean(((trade_sizes - avg_trade_size) / size_std) ** 3) if size_std > 0 else 0
    else:
        size_std = 0
        size_skewness = 0
    
    # Trade clustering
    if len(time_diffs) > 0:
        time_diff_std = time_diffs.std()
        clustering = time_diff_std / avg_time_between_trades if avg_time_between_trades > 0 else 0
    else:
        time_diff_std = 0
        clustering = 0
    
    # Momentum features
    recent_trades = df_trades.tail(10)  # Last 10 trades
    if len(recent_trades) > 1:
        recent_price_changes = recent_trades['price'].diff().dropna()
        recent_momentum = recent_price_changes.mean() if len(recent_price_changes) > 0 else 0
    else:
        recent_momentum = 0
    
    # Trade intensity
    time_window = (df_trades['timestamp'].max() - df_trades['timestamp'].min()).total_seconds()
    trade_intensity = trade_count / time_window if time_window > 0 else 0
    
    return {
        'trade_volume': total_volume,
        'trade_count': trade_count,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'volume_imbalance': volume_imbalance,
        'avg_price_change': avg_price_change,
        'price_volatility': price_volatility,
        'avg_trade_size': avg_trade_size,
        'large_trades': large_trades,
        'avg_time_between_trades': avg_time_between_trades,
        'trade_frequency': trade_frequency,
        'size_std': size_std,
        'size_skewness': size_skewness,
        'time_diff_std': time_diff_std,
        'clustering': clustering,
        'recent_momentum': recent_momentum,
        'trade_intensity': trade_intensity
    }

class SignalGenerator:
    """Generate trading signals based on order book and trade data"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.orderbook_history = deque(maxlen=window_size)
        self.trade_history = deque(maxlen=window_size)
        
    def update(self, orderbook_features, trade_features):
        """Update history with new data"""
        self.orderbook_history.append(orderbook_features)
        self.trade_history.append(trade_features)
        
    def generate_signal(self):
        """Generate trading signal based on order book and trade imbalances"""
        if len(self.orderbook_history) < self.window_size:
            return 0  # Hold signal when insufficient data
        
        # Calculate recent averages
        recent_depth_imbalance = np.mean([ob['depth_imbalance'] for ob in self.orderbook_history])
        recent_volume_imbalance = np.mean([tf['volume_imbalance'] for tf in self.trade_history])
        recent_large_orders = np.mean([ob['large_bids'] - ob['large_asks'] for ob in self.orderbook_history])
        
        # Signal logic based on imbalances
        signal = 0  # Default: hold
        
        # Long signal conditions
        if (recent_depth_imbalance > 0.3 and  # Strong bid dominance
            recent_volume_imbalance > 0.2 and  # More buying volume
            recent_large_orders > 1):          # More large buy orders
            signal = 1  # Buy signal
        
        # Short signal conditions
        elif (recent_depth_imbalance < -0.3 and  # Strong ask dominance
              recent_volume_imbalance < -0.2 and # More selling volume
              recent_large_orders < -1):         # More large sell orders
            signal = 2  # Sell signal
            
        return signal

class MarketMicrostructureTransformer(nn.Module):
    """Transformer model for market microstructure analysis"""
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(MarketMicrostructureTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        # Take the last timestep output
        x = x[:, -1, :]
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

def create_market_microstructure_dataset(orderbook_data, trade_data, sequence_length=20):
    """Create dataset for market microstructure transformer model"""
    # This is a simplified implementation
    # In practice, you would create sequences of order book and trade data
    # For now, we'll just return placeholder data
    return None

async def list_available_symbols(exchange):
    try:
        await exchange.load_markets()
        logging.debug("Рынки загружены")
        return exchange.symbols
    except Exception as e:
        logging.error(f"Ошибка при загрузке рынков: {e}")
        return []

async def verify_symbol(exchange, symbol):
    try:
        await exchange.load_markets()
        is_valid = symbol in exchange.symbols
        logging.debug(f"Проверка символа {symbol}: {'доступен' if is_valid else 'недоступен'}")
        return is_valid
    except Exception as e:
        logging.error(f"Ошибка при проверке символа: {e}")
        return False

async def get_full_data_with_features(exchange, symbol, timeframe='1s', since=None, limit=2000):
    """Get full data including order book and trade features for training"""
    all_data = []
    logging.info(f"Начало получения данных с признаками для символа {symbol}")
    
    # Fetch initial data
    ohlcv_data = await get_full_data(exchange, symbol, timeframe, since, limit)
    if ohlcv_data is None or ohlcv_data.empty:
        return None
    
    # For each OHLCV data point, fetch corresponding order book and trade data
    for idx, row in ohlcv_data.iterrows():
        try:
            # Fetch order book data
            orderbook = await get_order_book_data(exchange, symbol, limit=50)
            if not orderbook:
                logging.warning(f"Не удалось получить данные стакана для точки {idx}")
                continue
            
            # Fetch trade data
            trades = await get_trade_data(exchange, symbol, limit=100)
            if trades is None:
                logging.warning(f"Не удалось получить данные ленты сделок для точки {idx}")
                continue
            
            # Calculate features
            ob_features = calculate_order_book_features(orderbook)
            trade_features = calculate_trade_features(trades)
            
            if not ob_features or not trade_features:
                logging.warning(f"Не удалось рассчитать признаки для точки {idx}")
                continue
            
            # Combine all data into one row
            combined_row = row.to_dict()
            combined_row.update(ob_features)
            combined_row.update(trade_features)
            
            all_data.append(combined_row)
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Ошибка при обработке точки {idx}: {e}")
            continue
    
    if not all_data:
        logging.error("Не удалось получить данные с признаками")
        return None
        
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logging.info(f"Получено {len(df)} записей данных с признаками для символа {symbol}")
    return df

def get_or_train_model_sync(symbol, train_df, models_dir, best_params=None, force_train=False):
    logging.info(f"Получение или обучение модели для символа {symbol}")
    model_path = f'{models_dir}/{symbol.replace("/", "_").replace(":", "_")}_ppo'
    norm_path = f'{model_path}_norm.json'
    env = TradingEnvironment(train_df)
    norm_params = {'means': env.means.to_dict(), 'stds': env.stds.to_dict()}
    
    if os.path.exists(f"{model_path}.zip") and not force_train:
        logging.info("Нахождение существующей модели, загрузка модели")
        if os.path.exists(norm_path):
            with open(norm_path, 'r') as f:
                norm_params = json.load(f)
        env = TradingEnvironment(train_df, norm_params=norm_params)
        env = DummyVecEnv([lambda: env])
        model = PPO.load(model_path, env=env)
        logging.info("Модель загружена успешно")
    else:
        if force_train:
            logging.info("Запуск принудительного переобучения модели")
        else:
            logging.info("Модель не найдена, начало обучения")
        env = TradingEnvironment(train_df)
        means = env.means.to_dict()
        stds = env.stds.to_dict()
        env = DummyVecEnv([lambda: env])
        
        # ✅ УЛУЧШЕНО: Увеличены timesteps для лучшего качества модели
        if force_train:
            # Качественное переобучение для HFT
            total_timesteps = 100_000  # Было 20K → Теперь 100K (~2-3 мин)
            logging.info("🔄 Переобучение HFT-модели (100K шагов, ~2-3 мин)")
        else:
            # Глубокое начальное обучение для HFT
            total_timesteps = 500_000  # Было 50K → Теперь 500K (~10-15 мин)
            logging.info("🎯 Первичное обучение HFT-модели (500K шагов, ~10-15 мин)")
        callback = ProgressCallback(total_timesteps)
        
        if best_params:
            net_arch = []
            n_layers = best_params.get('n_layers', 1)
            for i in range(n_layers):
                layer_size = best_params.get(f'n_units_l{i}', 64)
                net_arch.append(layer_size)
            activation = best_params.get('activation', 'tanh')
            activation_mapping = {
                'relu': torch.nn.ReLU,
                'tanh': torch.nn.Tanh,
                'elu': torch.nn.ELU
            }
            activation_fn = activation_mapping.get(activation, torch.nn.Tanh)
            policy_kwargs = dict(
                net_arch=dict(pi=net_arch, vf=net_arch),
                activation_fn=activation_fn
            )
            model = PPO('MlpPolicy',
                        env,
                        learning_rate=best_params['learning_rate'],
                        n_steps=best_params['n_steps'],
                        gamma=best_params['gamma'],
                        ent_coef=best_params['ent_coef'],
                        vf_coef=best_params['vf_coef'],
                        max_grad_norm=best_params['max_grad_norm'],
                        policy_kwargs=policy_kwargs,
                        tensorboard_log="./ppo_tensorboard/",
                        verbose=1)
        else:
            model = PPO('MlpPolicy', env, tensorboard_log="./ppo_tensorboard/", verbose=1)
            
        logging.info("Запуск обучения модели...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        logging.info("Обучение модели завершено (100%)")
        
        model.save(model_path)
        with open(norm_path, 'w') as f:
            json.dump(norm_params, f)
        logging.info("Модель обучена и сохранена")
    return model, norm_params

def backtest_model_sync(model, test_df, symbol, norm_params):
    logging.info(f"Начало бэктеста модели для символа {symbol}")
    test_env = TradingEnvironment(test_df, norm_params=norm_params)
    obs, _ = test_env.reset()
    while not test_env.done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
    logging.info("Бэктест завершен")

def objective_sync(trial, train_df, test_df):
    try:
        logging.debug(f"Начало оптимизации trial {trial.number}")
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical('n_steps', [128, 256, 512])
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
        vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 5.0)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        net_arch = []
        for i in range(n_layers):
            layer_size = trial.suggest_int(f'n_units_l{i}', 64, 512)
            net_arch.append(layer_size)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu', 'elu'])
        activation_mapping = {
            'relu': torch.nn.ReLU,
            'tanh': torch.nn.Tanh,
            'elu': torch.nn.ELU
        }
        activation_fn = activation_mapping.get(activation, torch.nn.Tanh)
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            activation_fn=activation_fn
        )
        env = TradingEnvironment(train_df)
        means = env.means.to_dict()
        stds = env.stds.to_dict()
        env = DummyVecEnv([lambda: env])
        model = PPO('MlpPolicy',
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    gamma=gamma,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./ppo_tensorboard/",
                    verbose=0)
        model.learn(total_timesteps=100000)
        test_env = TradingEnvironment(test_df, norm_params={'means': means, 'stds': stds})
        obs, _ = test_env.reset()
        total_reward = 0
        while not test_env.done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            total_reward += reward
        env.close()
        test_env.close()
        logging.debug(f"Trial {trial.number} завершен с наградой {total_reward}")
        return total_reward
    except Exception as e:
        logging.error(f"Ошибка в trial {trial.number}: {e}")
        return float('-inf')

async def run_optuna(study, train_df, test_df, n_trials):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor()
    for _ in range(n_trials):
        trial = study.ask()
        score = await loop.run_in_executor(executor, objective_sync, trial, train_df, test_df)
        study.tell(trial, score)
    executor.shutdown(wait=True)

def get_real_balance_sync(exchange):
    try:
        balance = asyncio.run(exchange.fetch_balance())
        real_balance = balance['total'].get('USDT', 0)
        logging.debug(f"Текущий баланс: {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Ошибка при получении баланса: {e}")
        return None

class LiveTradingState:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        self.balance_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.balance = None

    def update(self, new_row, current_balance, timestamp):
        self.data.append(new_row)
        self.balance_history.append(current_balance)
        self.timestamps.append(timestamp)
        self.balance = current_balance
        logging.debug("Состояние live_trading обновлено")

    def get_dataframe(self):
        if len(self.data) == self.window_size:
            df = pd.DataFrame(list(self.data))
            df['balance'] = list(self.balance_history)
            df['timestamp'] = pd.to_datetime(list(self.timestamps))
            logging.debug("DataFrame для live_trading готов")
            return df
        else:
            logging.debug("Недостаточно данных для формирования DataFrame")
            return None

async def get_real_balance_async(exchange):
    try:
        # For demo accounts, we'll use a fixed balance
        # In a real account, this would fetch the actual balance
        if exchange.options.get('demo', False):
            # Return a fixed demo balance
            return 10000.0  # 10,000 USDT demo balance
        
        balance = await exchange.fetch_balance()
        real_balance = balance['total'].get('USDT', 0)
        logging.debug(f"Текущий баланс (асинхронно): {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Ошибка при получении баланса: {e}")
        # For demo purposes, return a default balance if there's an error
        if exchange.options.get('demo', False):
            return 10000.0
        return None

async def live_trading(async_exchange, model, symbol, norm_params, state):
    trading_interval = 1  # Changed to 1 second for high-frequency trading
    logging.info("Запуск live_trading")
    while True:
        try:
            real_balance = await get_real_balance_async(async_exchange)
            logging.debug(f"Текущий баланс: {real_balance}")
            if real_balance is None:
                logging.warning("Не удалось получить баланс, ожидание перед следующей попыткой")
                await asyncio.sleep(trading_interval)
                continue
            
            # Fetch order book data
            orderbook = await get_order_book_data(async_exchange, symbol, limit=50)
            if not orderbook:
                logging.warning("Не удалось получить данные стакана")
                await asyncio.sleep(trading_interval)
                continue
            
            # Fetch trade data
            trades = await get_trade_data(async_exchange, symbol, limit=100)
            if trades is None:
                logging.warning("Не удалось получить данные ленты сделок")
                await asyncio.sleep(trading_interval)
                continue
            
            # Calculate order book features
            ob_features = calculate_order_book_features(orderbook)
            if not ob_features:
                logging.warning("Не удалось рассчитать признаки стакана")
                await asyncio.sleep(trading_interval)
                continue
            
            # Calculate trade features
            trade_features = calculate_trade_features(trades)
            if not trade_features:
                logging.warning("Не удалось рассчитать признаки ленты сделок")
                await asyncio.sleep(trading_interval)
                continue
            
            # Fetch OHLCV data
            ohlcv = await async_exchange.fetch_ohlcv(symbol, timeframe='1s', limit=1)
            if not ohlcv:
                logging.warning("Не удалось получить новые данные OHLCV")
                await asyncio.sleep(trading_interval)
                continue
            
            df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
            
            # Add ATR approximation using order book spread
            df_new['atr'] = ob_features.get('spread', 0.001)
            
            # Combine all features
            for key, value in ob_features.items():
                df_new[key] = value
            for key, value in trade_features.items():
                df_new[key] = value
            
            timestamp = df_new['timestamp'].iloc[0]
            state.update(df_new.iloc[0], real_balance, timestamp)
            df = state.get_dataframe()
            if df is None:
                await asyncio.sleep(trading_interval)
                continue
            
            # Remove technical indicators calculation
            # df = add_technical_indicators(df)
            
            if not all(feat in df.columns for feat in norm_params['means'].keys()):
                logging.warning("Недостающие признаки после добавления индикаторов")
                await asyncio.sleep(trading_interval)
                continue
            
            try:
                normalized_df = (df.drop(columns=['timestamp', 'balance']) - pd.Series(norm_params['means'])) / pd.Series(norm_params['stds'])
                obs = normalized_df.values.flatten().astype(np.float32)
                logging.debug(f"Наблюдение сформировано: {obs.shape}")
            except Exception as e:
                logging.error(f"Ошибка при нормализации данных: {e}")
                await asyncio.sleep(trading_interval)
                continue
            
            if obs.shape[0] != model.observation_space.shape[0]:
                logging.error(f"Неожиданная форма наблюдения {obs.shape}, ожидается {model.observation_space.shape}")
                await asyncio.sleep(trading_interval)
                continue
            
            action, _states = model.predict(obs, deterministic=True)
            logging.debug(f"Предсказанное действие: {action}")
            
            positions = await async_exchange.fetch_positions(symbol)
            has_position = False
            current_contracts = 0
            current_side = None
            entry_price = 0
            if positions and isinstance(positions, list):
                for position in positions:
                    if position and 'contracts' in position and float(position.get('contracts', 0)) > 0:
                        has_position = True
                        current_contracts = float(position['contracts'])
                        current_side = position.get('side', '').lower()
                        entry_price = float(position.get('entryPrice', 0))
                        break
            
            current_price = float(df['close'].iloc[-1])
            amount = real_balance * 0.01 / current_price  # Changed to 1% risk
            # Use order book spread for ATR approximation
            atr = ob_features.get('spread', 0.001)
            
            if action == 1:
                if has_position and current_side in ['sell', 'short']:
                    close_side = 'buy'
                    logging.info("Переключение с шорт на лонг: закрытие текущей позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side=close_side, amount=current_contracts)
                if not has_position or current_side != 'long':
                    logging.info("Открытие длинной позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side='buy', amount=amount)
            elif action == 2:
                if has_position and current_side in ['buy', 'long']:
                    close_side = 'sell'
                    logging.info("Переключение с лонг на шорт: закрытие текущей позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side=close_side, amount=current_contracts)
                if not has_position or current_side != 'short':
                    logging.info("Открытие короткой позиции")
                    order = await async_exchange.create_order(symbol=symbol, type='market', side='sell', amount=amount)
            # Действие 0: удерживать позицию, никаких действий не требуется
        except Exception as e:
            logging.error(f"Ошибка в live_trading: {e}")
        await asyncio.sleep(trading_interval)

async def live_trading_multi_pair(async_exchange, models, symbols, norm_params_dict, states, tick_counters, historical_data, ticks_for_training, models_dir, executor):
    """Live trading function for multiple pairs with retraining"""
    trading_interval = 5  # 5 seconds for readable logs
    logging.info("Запуск многопарной live_trading")
    
    while True:
        try:
            real_balance = await get_real_balance_async(async_exchange)
            logging.debug(f"Текущий баланс: {real_balance}")
            if real_balance is None:
                logging.warning("Не удалось получить баланс, ожидание перед следующей попыткой")
                await asyncio.sleep(trading_interval)
                continue
                
            # Process each symbol
            for symbol in symbols:
                try:
                    # Check if model is available
                    if symbol not in models:
                        continue
                        
                    model = models[symbol]
                    norm_params = norm_params_dict[symbol]
                    state = states[symbol]
                    
                    # Fetch order book data
                    orderbook = await get_order_book_data(async_exchange, symbol, limit=50)
                    if not orderbook:
                        logging.warning(f"Не удалось получить данные стакана для {symbol}")
                        continue
                    
                    # Fetch trade data
                    trades = await get_trade_data(async_exchange, symbol, limit=100)
                    if trades is None:
                        logging.warning(f"Не удалось получить данные ленты сделок для {symbol}")
                        continue
                    
                    # Calculate order book features
                    ob_features = calculate_order_book_features(orderbook)
                    if not ob_features:
                        logging.warning(f"Не удалось рассчитать признаки стакана для {symbol}")
                        continue
                    
                    # Calculate trade features
                    trade_features = calculate_trade_features(trades)
                    if not trade_features:
                        logging.warning(f"Не удалось рассчитать признаки ленты сделок для {symbol}")
                        continue
                    
                    # Fetch OHLCV data
                    ohlcv = await async_exchange.fetch_ohlcv(symbol, timeframe='1s', limit=1)
                    if not ohlcv:
                        logging.warning(f"Не удалось получить новые данные OHLCV для {symbol}")
                        continue
                    
                    df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
                    
                    # Add ATR approximation using order book spread
                    df_new['atr'] = ob_features.get('spread', 0.001)
                    
                    # Combine all features
                    for key, value in ob_features.items():
                        df_new[key] = value
                    for key, value in trade_features.items():
                        df_new[key] = value
                    
                    timestamp = df_new['timestamp'].iloc[0]
                    state.update(df_new.iloc[0], real_balance, timestamp)
                    df = state.get_dataframe()
                    if df is None:
                        continue
                    
                    if not all(feat in df.columns for feat in norm_params['means'].keys()):
                        logging.warning(f"Недостающие признаки после добавления индикаторов для {symbol}")
                        continue
                    
                    # Store the data record for retraining
                    if symbol in historical_data:
                        historical_data[symbol].append(df_new.iloc[0].to_dict())
                        tick_counters[symbol] += 1
                        
                        # Keep only the last N records to prevent memory issues
                        if len(historical_data[symbol]) > ticks_for_training * 5:
                            historical_data[symbol] = historical_data[symbol][-ticks_for_training*2:]
                            
                    # Check for retraining
                    if tick_counters[symbol] >= ticks_for_training:
                        logging.info(f"Накоплено {tick_counters[symbol]} тиков для {symbol}. Начало переобучения...")
                        
                        # Prepare data for retraining
                        df = pd.DataFrame(historical_data[symbol])
                        train_size = int(len(df) * 0.8)
                        if train_size >= 10:
                            train_df = df.iloc[:train_size].reset_index(drop=True)
                            
                            # Retrain model
                            loop = asyncio.get_running_loop()
                            symbol_model_dir = os.path.join(models_dir, symbol.replace("/", "_").replace(":", "_"))
                            
                            # We use the existing best params if available, or None (which defaults to standard PPO)
                            # In a full implementation, we might want to re-optimize hyperparameters occasionally
                            model, norm_params = await loop.run_in_executor(
                                executor, 
                                get_or_train_model_sync, 
                                symbol, 
                                train_df, 
                                symbol_model_dir, 
                                None, # best_params
                                True  # force_train
                            )
                            
                            # Update model and params
                            models[symbol] = model
                            norm_params_dict[symbol] = norm_params
                            logging.info(f"Модель для {symbol} успешно переобучена. Прогресс: 100%")
                            
                            # Reset counter
                            tick_counters[symbol] = 0
                        else:
                            logging.warning(f"Недостаточно данных для переобучения {symbol}")
                    
                    try:
                        normalized_df = (df.drop(columns=['timestamp', 'balance']) - pd.Series(norm_params['means'])) / pd.Series(norm_params['stds'])
                        obs = normalized_df.values.flatten().astype(np.float32)
                        logging.debug(f"Наблюдение сформировано для {symbol}: {obs.shape}")
                    except Exception as e:
                        logging.error(f"Ошибка при нормализации данных для {symbol}: {e}")
                        continue
                    
                    if obs.shape[0] != model.observation_space.shape[0]:
                        logging.error(f"Неожиданная форма наблюдения {obs.shape} для {symbol}, ожидается {model.observation_space.shape}")
                        continue
                    
                    action, _states = model.predict(obs, deterministic=True)
                    logging.debug(f"Предсказанное действие для {symbol}: {action}")
                    
                    positions = await async_exchange.fetch_positions(symbol)
                    has_position = False
                    current_contracts = 0
                    current_side = None
                    entry_price = 0
                    if positions and isinstance(positions, list):
                        for position in positions:
                            if position and 'contracts' in position and float(position.get('contracts', 0)) > 0:
                                has_position = True
                                current_contracts = float(position['contracts'])
                                current_side = position.get('side', '').lower()
                                entry_price = float(position.get('entryPrice', 0))
                                break
                    
                    current_price = float(df['close'].iloc[-1])
                    amount = real_balance * 0.01 / current_price  # Changed to 1% risk
                    # Use order book spread for ATR approximation
                    atr = ob_features.get('spread', 0.001)
                    
                    if action == 1:
                        if has_position and current_side in ['sell', 'short']:
                            close_side = 'buy'
                            logging.info(f"Переключение с шорт на лонг для {symbol}: закрытие текущей позиции")
                            order = await async_exchange.create_order(symbol=symbol, type='market', side=close_side, amount=current_contracts)
                        if not has_position or current_side != 'long':
                            logging.info(f"Открытие длинной позиции для {symbol}")
                            order = await async_exchange.create_order(symbol=symbol, type='market', side='buy', amount=amount)
                    elif action == 2:
                        if has_position and current_side in ['buy', 'long']:
                            close_side = 'sell'
                            logging.info(f"Переключение с лонг на шорт для {symbol}: закрытие текущей позиции")
                            order = await async_exchange.create_order(symbol=symbol, type='market', side=close_side, amount=current_contracts)
                        if not has_position or current_side != 'short':
                            logging.info(f"Открытие короткой позиции для {symbol}")
                            order = await async_exchange.create_order(symbol=symbol, type='market', side='sell', amount=amount)
                    # Действие 0: удерживать позицию, никаких действий не требуется
                except Exception as e:
                    logging.error(f"Ошибка в live_trading для {symbol}: {e}", exc_info=True)
                    continue
                    
            # Portfolio status report after each cycle
            try:
                total_positions = sum(1 for s in symbols if s in models)
                portfolio_status = f"\n{'='*60}\n"
                portfolio_status += f"📊 СОСТОЯНИЕ ПОРТФЕЛЯ\n"
                portfolio_status += f"{'='*60}\n"
                portfolio_status += f"💰 Баланс: {real_balance:.2f} USDT\n"
                portfolio_status += f"📈 Активных моделей: {total_positions}\n"
                portfolio_status += f"🔄 Тиков собрано: {tick_counters.get('BTC/USDT:USDT', 0)}/{ticks_for_training}\n"
                portfolio_status += f"{'='*60}\n"
                logging.info(portfolio_status)
            except Exception as e:
                logging.error(f"Ошибка при выводе статуса портфеля: {e}")
                    
            await asyncio.sleep(trading_interval)
        except Exception as e:
            logging.error(f"Ошибка в многопарной live_trading: {e}")
            await asyncio.sleep(trading_interval)

def shutdown_handler():
    logging.info("Обработка сигнала завершения")
    for task in asyncio.all_tasks():
        task.cancel()

async def main():
    loop = asyncio.get_running_loop()
    if not sys.platform.startswith('win'):
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
    async_exchange = ccxt_async.bybit(exchange_config)
    # Load markets to synchronize time
    try:
        await async_exchange.load_markets()
        logging.info("Рынки успешно загружены, время синхронизировано")
    except Exception as e:
        logging.error(f"Ошибка при синхронизации времени: {e}")
    executor = ThreadPoolExecutor()
    try:
        # List of liquid trading pairs
        symbols = [
            "BTC/USDT:USDT"
        ]
        
        # Number of ticks to collect before starting training
        ticks_for_training = 50  # Balanced for stable training
        
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Verify all symbols are available
        available_symbols = await list_available_symbols(async_exchange)
        valid_symbols = [symbol for symbol in symbols if symbol in available_symbols]
        
        if not valid_symbols:
            logging.error("Ни одна из указанных торговых пар недоступна")
            return
            
        logging.info(f"Доступные торговые пары: {valid_symbols}")
        
        # Dictionaries to store models, parameters and states for each symbol
        models = {}
        norm_params_dict = {}
        states = {}
        tick_counters = {symbol: 0 for symbol in valid_symbols}
        historical_data = {symbol: [] for symbol in valid_symbols}
        
        # Process each symbol
        for symbol in valid_symbols:
            try:
                logging.info(f"Обработка торговой пары: {symbol}")
                
                if not await verify_symbol(async_exchange, symbol):
                    logging.warning(f"Символ {symbol} недоступен, пропуск")
                    continue
                    
                # Create state for this symbol
                state = LiveTradingState(window_size=20)
                states[symbol] = state
                
                # Initialize tick counter and historical data storage for this symbol
                tick_counters[symbol] = 0
                historical_data[symbol] = []
                
            except Exception as e:
                logging.error(f"Ошибка при инициализации символа {symbol}: {e}")
                continue
                
        # Start collecting data and training when enough ticks are accumulated
        logging.info(f"Начало сбора данных. Обучение начнется после накопления {ticks_for_training} тиков для каждой пары")
        await collect_and_train_data(async_exchange, valid_symbols, models, norm_params_dict, states, tick_counters, historical_data, ticks_for_training, models_dir, executor)
                
    except asyncio.CancelledError:
        logging.info("Задачи были отменены")
    except Exception as e:
        logging.error(f"Ошибка в main: {e}")
    finally:
        await async_exchange.close()
        executor.shutdown(wait=True)
        logging.info("Обмен закрыт и исполнитель завершен")

async def collect_and_train_data(async_exchange, symbols, models, norm_params_dict, states, tick_counters, historical_data, ticks_for_training, models_dir, executor):
    """Collect data and start training when enough ticks are accumulated"""
    trading_interval = 1  # 1 second for high-frequency data collection
    
    while True:
        try:
            # Check if we have enough data for all symbols to start training
            all_symbols_ready = all(tick_counters[symbol] >= ticks_for_training for symbol in symbols if symbol in tick_counters)
            
            if all_symbols_ready and not models:
                logging.info("Достаточно тиков накоплено для всех пар. Начало обучения моделей.")
                await train_models(symbols, historical_data, models, norm_params_dict, states, models_dir, executor)
                logging.info("Модели обучены. Переход к реальной торговле.")
                # After training, start live trading
                await live_trading_multi_pair(async_exchange, models, symbols, norm_params_dict, states, tick_counters, historical_data, ticks_for_training, models_dir, executor)
                return  # Exit data collection loop after starting live trading
                
            # Collect data for each symbol
            for symbol in symbols:
                try:
                    if symbol not in tick_counters:
                        continue
                        
                    # Fetch order book data
                    orderbook = await get_order_book_data(async_exchange, symbol, limit=50)
                    if not orderbook:
                        logging.warning(f"Не удалось получить данные стакана для {symbol}")
                        continue
                    
                    # Fetch trade data
                    trades = await get_trade_data(async_exchange, symbol, limit=100)
                    if trades is None:
                        logging.warning(f"Не удалось получить данные ленты сделок для {symbol}")
                        continue
                    
                    # Calculate order book features
                    ob_features = calculate_order_book_features(orderbook)
                    if not ob_features:
                        logging.warning(f"Не удалось рассчитать признаки стакана для {symbol}")
                        continue
                    
                    # Calculate trade features
                    trade_features = calculate_trade_features(trades)
                    if not trade_features:
                        logging.warning(f"Не удалось рассчитать признаки ленты сделок для {symbol}")
                        continue
                    
                    # Fetch OHLCV data
                    ohlcv = await async_exchange.fetch_ohlcv(symbol, timeframe='1s', limit=1)
                    if not ohlcv:
                        logging.warning(f"Не удалось получить новые данные OHLCV для {symbol}")
                        continue
                    
                    # Combine all features into a single record
                    df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
                    
                    # Add order book and trade features
                    for key, value in ob_features.items():
                        df_new[key] = value
                    for key, value in trade_features.items():
                        df_new[key] = value
                    
                    # Add ATR approximation using order book spread
                    df_new['atr'] = ob_features.get('spread', 0.001)
                    
                    # Store the data record
                    if symbol in historical_data:
                        historical_data[symbol].append(df_new.iloc[0].to_dict())
                        tick_counters[symbol] += 1
                        
                        # Keep only the last N records to prevent memory issues
                        if len(historical_data[symbol]) > ticks_for_training * 2:
                            historical_data[symbol] = historical_data[symbol][-ticks_for_training:]
                    
                    if tick_counters[symbol] % 10 == 0:
                        logging.info(f"📊 Собрано {tick_counters[symbol]} тиков для {symbol}")
                    
                except Exception as e:
                    logging.error(f"Ошибка при сборе данных для {symbol}: {e}")
                    continue
                    
            await asyncio.sleep(trading_interval)
            
        except Exception as e:
            logging.error(f"Ошибка в цикле сбора данных: {e}")
            await asyncio.sleep(trading_interval)

async def train_models(symbols, historical_data, models, norm_params_dict, states, models_dir, executor):
    """Train models for each symbol using collected data"""
    loop = asyncio.get_running_loop()
    for symbol in symbols:
        try:
            logging.info(f"Данные для {symbol}: {len(historical_data.get(symbol, []))} записей")
            if symbol not in historical_data or len(historical_data[symbol]) < 40:  # Minimum data requirement
                logging.warning(f"Недостаточно данных для обучения модели {symbol}")
                continue
                
            logging.info(f"Начало обучения модели для {symbol}")
            
            # Convert historical data to DataFrame
            df = pd.DataFrame(historical_data[symbol])
            
            # Ensure we have required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logging.warning(f"Недостаточно данных для {symbol}")
                continue
                
            # Prepare data for training
            train_size = int(len(df) * 0.8)
            if train_size < 10:  # Minimum training size
                logging.warning(f"Недостаточно данных для разделения на обучающую и тестовую выборки для {symbol}")
                continue
                
            train_df = df.iloc[:train_size].reset_index(drop=True)
            test_df = df.iloc[train_size:].reset_index(drop=True)
            
            # Create symbol-specific model directory
            symbol_model_dir = os.path.join(models_dir, symbol.replace("/", "_").replace(":", "_"))
            os.makedirs(symbol_model_dir, exist_ok=True)
            
            # ✅ УЛУЧШЕНО: Качественная оптимизация гиперпараметров для HFT
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10,  # Минимум trials перед pruning
                    n_warmup_steps=5,      # Шагов перед оценкой
                    interval_steps=3       # Интервал проверки
                ),
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=10,
                    multivariate=True
                )
            )
            # ✅ Увеличено с 3 до 50 trials для качественной оптимизации
            await run_optuna(study, train_df, test_df, n_trials=50)
            best_params = study.best_params
            logging.info(f"🎯 Лучшие параметры оптимизации для {symbol}: {best_params}")
            
            model, norm_params = await loop.run_in_executor(executor, get_or_train_model_sync, symbol, train_df, symbol_model_dir, best_params)
            await loop.run_in_executor(executor, backtest_model_sync, model, test_df, symbol, norm_params)
            
            # Store model and parameters
            models[symbol] = model
            norm_params_dict[symbol] = norm_params
            
            # Initialize state with recent data
            last_20_data = test_df.tail(20)
            initial_balance = 10  # Default balance for initialization
            for _, row in last_20_data.iterrows():
                states[symbol].update(row, initial_balance, row['timestamp'])
                
            logging.info(f"Модель для {symbol} успешно обучена")
            
        except Exception as e:
            logging.error(f"Ошибка при обучении модели для {symbol}: {e}")
            continue

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Программа прервана пользователем")
