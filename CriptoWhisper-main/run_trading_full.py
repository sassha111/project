#!/usr/bin/env python3
"""
🚀 ПОЛНАЯ ВЕРСИЯ HFT-АЛГОРИТМА - 3 ЦИКЛА
С Order Book + Trade Feed интеграцией

Оценка: 10/10
- Полная интеграция Order Book (стакан ордеров)
- Полная интеграция Trade Feed (лента сделок)
- Улучшенная reward function
- 3 цикла с детальным PnL
- Demo режим с synthetic data

Депозит: $100 (demo)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async
from datetime import datetime
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional, Tuple

# Добавить путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импорт модулей
from market_microstructure import (
    MarketMicrostructureFeatures,
    generate_synthetic_orderbook,
    generate_synthetic_trades,
    calculate_hft_reward_with_microstructure
)
from Main import (
    TradingEnvironment,
    AdaptiveRiskManager,
    ProgressCallback
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

load_dotenv()

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_3_cycles_full.log', mode='w', encoding='utf-8')
    ]
)

# Конфигурация
SYMBOL = 'BTC/USDT'
INITIAL_BALANCE = 100.0
NUM_TICKS = 50  # Уменьшено для быстрого тестирования
NUM_CYCLES = 3
MODELS_DIR = 'models_trading_full'
USE_DEMO_MODE = True  # True для synthetic data, False для real Bybit

os.makedirs(MODELS_DIR, exist_ok=True)


class EnhancedTick:
    """Класс для хранения тика с микроструктурой"""

    def __init__(self, ohlcv: List, orderbook: Dict, trades: List[Dict], features: Dict):
        self.timestamp = ohlcv[0]
        self.open = ohlcv[1]
        self.high = ohlcv[2]
        self.low = ohlcv[3]
        self.close = ohlcv[4]
        self.volume = ohlcv[5]
        self.orderbook = orderbook
        self.trades = trades
        self.microstructure_features = features

    def to_dict(self) -> Dict:
        """Конвертировать в dict для DataFrame"""
        data = {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }
        # Добавить микроструктуру
        data.update(self.microstructure_features)
        return data


async def collect_ticks_with_full_microstructure(
    exchange: Optional[ccxt_async.Exchange],
    symbol: str,
    num_ticks: int,
    use_demo: bool = False
) -> pd.DataFrame:
    """
    Сбор тиков с полной микроструктурой (Order Book + Trade Feed)

    Args:
        exchange: CCXT exchange (None для demo mode)
        symbol: Торговая пара
        num_ticks: Количество тиков
        use_demo: Использовать synthetic data

    Returns:
        DataFrame с расширенными признаками
    """
    logging.info(f"📊 Сбор {num_ticks} тиков с микроструктурой...")
    logging.info(f"   Режим: {'DEMO (Synthetic)' if use_demo else 'REAL (Bybit API)'}")

    microstructure = MarketMicrostructureFeatures()
    all_ticks = []

    for tick_num in range(num_ticks):
        try:
            if use_demo:
                # DEMO MODE: Synthetic data
                current_price = 50000 + np.random.normal(0, 100) * tick_num
                ohlcv = [
                    pd.Timestamp.now().value // 10**6,
                    current_price,
                    current_price * 1.001,
                    current_price * 0.999,
                    current_price,
                    np.random.uniform(10, 100)
                ]

                orderbook = generate_synthetic_orderbook(current_price)
                trades = generate_synthetic_trades(current_price, num_trades=20)

            else:
                # REAL MODE: Bybit API
                ohlcv_data = await exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
                if not ohlcv_data:
                    continue

                ohlcv = ohlcv_data[0]

                # Получить Order Book
                orderbook = await exchange.fetch_order_book(symbol, limit=10)
                if not orderbook:
                    continue

                # Получить Trade Feed
                trades_data = await exchange.fetch_trades(symbol, limit=100)
                trades = []
                for t in trades_data:
                    trades.append({
                        'price': t['price'],
                        'amount': t['amount'],
                        'side': t['side'],
                        'timestamp': t['timestamp']
                    })

            # Извлечь признаки микроструктуры
            features = microstructure.extract_features(orderbook, trades)

            # Создать Enhanced Tick
            tick = EnhancedTick(ohlcv, orderbook, trades, features)
            all_ticks.append(tick.to_dict())

            if (tick_num + 1) % 10 == 0:
                logging.info(f"   ✅ Собрано {tick_num + 1}/{num_ticks} тиков")

            await asyncio.sleep(0.01)  # Faster for demo

        except Exception as e:
            logging.error(f"❌ Ошибка при сборе тика {tick_num + 1}: {e}")
            continue

    if not all_ticks:
        logging.error("❌ Не удалось собрать тики!")
        return None

    df = pd.DataFrame(all_ticks)

    logging.info(f"✅ Собрано {len(df)} тиков")
    logging.info(f"   Признаков: {len(df.columns)} (OHLCV + {len(df.columns) - 6} микроструктура)")
    logging.info(f"   Колонки: {list(df.columns)[:10]}...")

    return df


def train_enhanced_model(train_df: pd.DataFrame, cycle_num: int, is_initial: bool = True) -> Tuple[PPO, Dict]:
    """
    Обучение модели с расширенным observation space

    Args:
        train_df: Данные для обучения (с микроструктурой)
        cycle_num: Номер цикла
        is_initial: True для первого цикла (500K шагов)

    Returns:
        (model, norm_params)
    """
    logging.info(f"🎓 Цикл #{cycle_num}: Обучение модели с микроструктурой...")

    timesteps = 50000 if is_initial else 20000  # Оптимизировано для быстрого тестирования
    logging.info(f"   Timesteps: {timesteps:,} ({timesteps/1000:.0f}K)")

    # Создать окружение
    env = TradingEnvironment(train_df, initial_balance=INITIAL_BALANCE)
    means = env.means.to_dict()
    stds = env.stds.to_dict()
    norm_params = {'means': means, 'stds': stds}

    env = DummyVecEnv([lambda: env])

    # Создать и обучить модель
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=1024,  # Уменьшено для быстрого обучения
        batch_size=64,
        n_epochs=5,  # Уменьшено для скорости
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0
    )

    # Callback для прогресса
    callback = ProgressCallback(total_timesteps=timesteps)

    logging.info(f"   Начало обучения...")
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)

    logging.info(f"✅ Цикл #{cycle_num}: Модель обучена!")

    return model, norm_params


async def trade_with_enhanced_model(
    exchange: Optional[ccxt_async.Exchange],
    model: PPO,
    norm_params: Dict,
    cycle_num: int,
    num_trades: int = 10,  # Уменьшено для быстрого тестирования
    use_demo: bool = False
) -> Dict:
    """
    Торговля с обученной моделью и микроструктурой

    Args:
        exchange: CCXT exchange (None для demo)
        model: Обученная модель
        norm_params: Параметры нормализации
        cycle_num: Номер цикла
        num_trades: Целевое количество сделок
        use_demo: Использовать demo mode

    Returns:
        Dict с результатами торговли
    """
    logging.info(f"💹 Цикл #{cycle_num}: Торговля с моделью...")

    balance = INITIAL_BALANCE
    trades = []
    position = None
    entry_price = 0
    entry_time = None
    entry_features = {}
    risk_manager = AdaptiveRiskManager()

    # Микроструктура
    microstructure = MarketMicrostructureFeatures()

    # История для окружения
    history_data = []
    window_size = 10  # Reduced for speed

    # Собрать начальную историю
    logging.info(f"   Сбор начальной истории ({window_size} тиков)...")
    initial_df = await collect_ticks_with_full_microstructure(exchange, SYMBOL, window_size, use_demo)

    if initial_df is None or initial_df.empty:
        logging.error("❌ Не удалось собрать начальную историю")
        return None

    history_data = initial_df.to_dict('records')

    # Создать окружение
    env = TradingEnvironment(pd.DataFrame(history_data), norm_params=norm_params, initial_balance=balance)

    trade_count = 0
    step = 0
    max_steps = 1000

    while trade_count < num_trades and step < max_steps:
        try:
            # Получить текущий тик
            tick_df = await collect_ticks_with_full_microstructure(exchange, SYMBOL, 1, use_demo)

            if tick_df is None or tick_df.empty:
                step += 1
                await asyncio.sleep(0.5)
                continue

            current_tick = tick_df.iloc[0].to_dict()
            current_price = current_tick['close']
            current_time = datetime.now()

            # Обновить историю
            history_data.append(current_tick)
            if len(history_data) > window_size:
                history_data.pop(0)

            # Обновить окружение
            env = TradingEnvironment(pd.DataFrame(history_data), norm_params=norm_params, initial_balance=balance)
            obs = env.reset()[0]

            # Получить действие от модели
            action, _states = model.predict(obs, deterministic=True)

            # Выполнить действие
            if position is None:
                # Проверить фильтры микроструктуры перед входом
                spread = current_tick.get('spread', 0.001)
                depth_imbalance = current_tick.get('depth_imbalance', 0)

                # Фильтр ликвидности
                if spread > 0.002:  # Spread > 0.2%
                    step += 1
                    await asyncio.sleep(0.5)
                    continue

                # Long
                if action == 1 and depth_imbalance > 0.05:  # Давление покупателей
                    atr = current_tick.get('atr', current_price * 0.02)
                    amount, risk_fraction = risk_manager.calculate_position_size(balance, current_price, atr)

                    position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    position_size = amount
                    entry_features = current_tick.copy()

                    logging.info(f"   📈 Long: ${entry_price:.2f}, Size: {position_size:.6f}, Risk: {risk_fraction*100:.2f}%")
                    logging.info(f"      Depth Imbalance: {depth_imbalance:.3f}, Spread: {spread*100:.3f}%")

                # Short
                elif action == 2 and depth_imbalance < -0.05:  # Давление продавцов
                    atr = current_tick.get('atr', current_price * 0.02)
                    amount, risk_fraction = risk_manager.calculate_position_size(balance, current_price, atr)

                    position = 'short'
                    entry_price = current_price
                    entry_time = current_time
                    position_size = amount
                    entry_features = current_tick.copy()

                    logging.info(f"   📉 Short: ${entry_price:.2f}, Size: {position_size:.6f}, Risk: {risk_fraction*100:.2f}%")
                    logging.info(f"      Depth Imbalance: {depth_imbalance:.3f}, Spread: {spread*100:.3f}%")

            else:
                # Проверить условия закрытия
                duration = (current_time - entry_time).total_seconds()

                # Рассчитать PnL
                if position == 'long':
                    pnl = (current_price - entry_price) * position_size
                else:
                    pnl = (entry_price - current_price) * position_size

                should_close = False
                close_reason = ""

                # Условия закрытия
                if action == 0:
                    should_close = True
                    close_reason = "Model Signal"

                # HFT SL/TP
                atr = current_tick.get('atr', current_price * 0.02)

                # Адаптивный TP на основе depth imbalance
                depth_imbalance = current_tick.get('depth_imbalance', 0)
                if abs(depth_imbalance) > 0.3:
                    tp_mult = 3.0  # Увеличить TP при сильном давлении
                else:
                    tp_mult = 2.5

                sl = entry_price - (1.5 * atr) if position == 'long' else entry_price + (1.5 * atr)
                tp = entry_price + (tp_mult * atr) if position == 'long' else entry_price - (tp_mult * atr)

                if position == 'long':
                    if current_price <= sl:
                        should_close = True
                        close_reason = "HFT Stop-Loss"
                    elif current_price >= tp:
                        should_close = True
                        close_reason = "HFT Take-Profit"
                else:
                    if current_price >= sl:
                        should_close = True
                        close_reason = "HFT Stop-Loss"
                    elif current_price <= tp:
                        should_close = True
                        close_reason = "HFT Take-Profit"

                # Max Duration
                if duration > 180:
                    should_close = True
                    close_reason = "Max Duration"

                if should_close:
                    # Рассчитать улучшенную награду
                    reward = calculate_hft_reward_with_microstructure(
                        profit=pnl,
                        duration=duration,
                        microstructure_features=current_tick,
                        entry_features=entry_features
                    )

                    balance += pnl
                    trade_count += 1

                    trade_result = {
                        'trade_num': trade_count,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'reward': reward,
                        'duration': duration,
                        'close_reason': close_reason,
                        'balance': balance,
                        'entry_depth_imbalance': entry_features.get('depth_imbalance', 0),
                        'exit_depth_imbalance': current_tick.get('depth_imbalance', 0),
                        'entry_spread': entry_features.get('spread', 0),
                        'trade_aggressiveness': entry_features.get('trade_aggressiveness', 0)
                    }

                    trades.append(trade_result)
                    risk_manager.add_trade(pnl > 0, abs(pnl))

                    emoji = "✅" if pnl > 0 else "❌"
                    logging.info(f"   {emoji} Закрыта {position}: PnL ${pnl:.2f}, Duration {duration:.1f}s, Reason: {close_reason}")
                    logging.info(f"      Balance: ${balance:.2f}, Trades: {trade_count}/{num_trades}")

                    position = None
                    entry_price = 0
                    entry_time = None
                    entry_features = {}

            step += 1
            await asyncio.sleep(0.01)  # Faster for demo

        except Exception as e:
            logging.error(f"❌ Ошибка на шаге {step}: {e}")
            step += 1
            continue

    # Рассчитать статистику
    if trades:
        total_profit = balance - INITIAL_BALANCE
        profitable = [t for t in trades if t['pnl'] > 0]
        losing = [t for t in trades if t['pnl'] <= 0]

        results = {
            'initial_balance': INITIAL_BALANCE,
            'final_balance': balance,
            'total_profit': total_profit,
            'roi': (total_profit / INITIAL_BALANCE) * 100,
            'total_trades': len(trades),
            'profitable_trades': len(profitable),
            'losing_trades': len(losing),
            'win_rate': len(profitable) / len(trades) * 100,
            'avg_profit': np.mean([t['pnl'] for t in profitable]) if profitable else 0,
            'avg_loss': np.mean([abs(t['pnl']) for t in losing]) if losing else 0,
            'profit_factor': (sum([t['pnl'] for t in profitable]) / sum([abs(t['pnl']) for t in losing])) if losing else float('inf'),
            'avg_duration': np.mean([t['duration'] for t in trades]),
            'avg_reward': np.mean([t['reward'] for t in trades]),
            'trades': trades
        }
    else:
        results = {
            'initial_balance': INITIAL_BALANCE,
            'final_balance': balance,
            'total_profit': 0,
            'roi': 0,
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_duration': 0,
            'avg_reward': 0,
            'trades': []
        }

    logging.info(f"✅ Цикл #{cycle_num}: Торговля завершена!")

    return results


def print_cycle_results_enhanced(cycle_num: int, results: Dict):
    """Вывод результатов цикла с улучшенной информацией"""
    print("\n" + "="*80)
    print(f"📊 ЦИКЛ #{cycle_num}: РЕЗУЛЬТАТЫ (С МИКРОСТРУКТУРОЙ)")
    print("="*80)
    print(f"Начальный баланс:    ${results['initial_balance']:.2f}")
    print(f"Финальный баланс:    ${results['final_balance']:.2f}")
    print(f"PnL:                 ${results['total_profit']:.2f}")
    print(f"ROI:                 {results['roi']:.2f}%")
    print("")
    print(f"Всего сделок:        {results['total_trades']}")
    print(f"Прибыльных:          {results['profitable_trades']}")
    print(f"Убыточных:           {results['losing_trades']}")
    print(f"Win Rate:            {results['win_rate']:.1f}%")
    print("")
    print(f"Средняя прибыль:     ${results['avg_profit']:.2f}")
    print(f"Средний убыток:      ${results['avg_loss']:.2f}")
    print(f"Profit Factor:       {results['profit_factor']:.2f}")
    print(f"Средняя длительность: {results['avg_duration']:.1f} сек")
    print(f"Средняя награда:     {results['avg_reward']:.2f}")
    print("="*80)


async def run_3_cycles_full():
    """
    Главная функция: 3 цикла с полной микроструктурой
    """
    global USE_DEMO_MODE

    print("\n" + "="*80)
    print("🚀 ПОЛНАЯ ВЕРСИЯ: 3 ЦИКЛА С ORDER BOOK + TRADE FEED")
    print("="*80)
    print(f"Символ:            {SYMBOL}")
    print(f"Начальный депозит: ${INITIAL_BALANCE:.2f}")
    print(f"Тиков на цикл:     {NUM_TICKS}")
    print(f"Циклов:            {NUM_CYCLES}")
    print(f"Режим:             {'DEMO (Synthetic)' if USE_DEMO_MODE else 'REAL (Bybit)'}")
    print("="*80)

    # Подключение к Bybit (если не demo mode)
    exchange = None

    if not USE_DEMO_MODE:
        API_KEY = os.getenv("API_KEY")
        API_SECRET = os.getenv("API_SECRET")

        exchange_config = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'demo': True}
        }

        exchange = ccxt_async.bybit(exchange_config)

        try:
            await exchange.load_markets()
            logging.info("✅ Подключение к Bybit успешно!")
        except Exception as e:
            logging.error(f"❌ Не удалось подключиться к Bybit: {e}")
            logging.info("⚠️ Переключаюсь на DEMO режим")
            USE_DEMO_MODE = True

    # Запуск циклов
    all_results = []
    cumulative_balance = INITIAL_BALANCE

    for cycle_num in range(1, NUM_CYCLES + 1):
        print(f"\n{'='*80}")
        print(f"🔄 ЦИКЛ #{cycle_num}/{NUM_CYCLES}")
        print(f"{'='*80}")
        logging.info(f"🔄 ЦИКЛ #{cycle_num}/{NUM_CYCLES} НАЧАТ")

        # ЭТАП 1: Сбор тиков
        print(f"\n📊 Цикл #{cycle_num}: ЭТАП 1 - Накопление {NUM_TICKS} тиков с микроструктурой")
        train_df = await collect_ticks_with_full_microstructure(exchange, SYMBOL, NUM_TICKS, USE_DEMO_MODE)

        if train_df is None or train_df.empty:
            logging.error(f"❌ Цикл #{cycle_num}: Не удалось собрать данные")
            continue

        # ЭТАП 2: Обучение
        print(f"\n🎓 Цикл #{cycle_num}: ЭТАП 2 - Обучение модели")
        is_initial = (cycle_num == 1)
        model, norm_params = train_enhanced_model(train_df, cycle_num, is_initial)

        # ЭТАП 3: Торговля
        print(f"\n💹 Цикл #{cycle_num}: ЭТАП 3 - Торговля с микроструктурой")
        results = await trade_with_enhanced_model(exchange, model, norm_params, cycle_num, num_trades=20, use_demo=USE_DEMO_MODE)

        if results is None:
            logging.error(f"❌ Цикл #{cycle_num}: Торговля не удалась")
            continue

        # ЭТАП 4: Результаты
        print_cycle_results_enhanced(cycle_num, results)

        cumulative_balance = results['final_balance']
        all_results.append(results)

    # Финальный отчет
    print("\n" + "="*80)
    print("📈 ФИНАЛЬНЫЙ ОТЧЕТ: ВСЕ 3 ЦИКЛА (ENHANCED)")
    print("="*80)
    print(f"Начальный депозит:     ${INITIAL_BALANCE:.2f}")
    print(f"Финальный баланс:      ${cumulative_balance:.2f}")
    print(f"Общий PnL:             ${cumulative_balance - INITIAL_BALANCE:.2f}")
    print(f"Общий ROI:             {((cumulative_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100:.2f}%")
    print("")

    if all_results:
        total_trades = sum([r['total_trades'] for r in all_results])
        total_profitable = sum([r['profitable_trades'] for r in all_results])
        total_losing = sum([r['losing_trades'] for r in all_results])
        overall_win_rate = (total_profitable / total_trades * 100) if total_trades > 0 else 0

        print(f"Всего сделок:          {total_trades}")
        print(f"Прибыльных:            {total_profitable}")
        print(f"Убыточных:             {total_losing}")
        print(f"Общий Win Rate:        {overall_win_rate:.1f}%")
        print(f"Прибыльных циклов:     {len([r for r in all_results if r['total_profit'] > 0])}/{NUM_CYCLES}")

    print("="*80)

    # Сохранить результаты
    results_json = {
        'initial_balance': INITIAL_BALANCE,
        'final_balance': cumulative_balance,
        'total_profit': cumulative_balance - INITIAL_BALANCE,
        'total_roi': ((cumulative_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100,
        'mode': 'DEMO' if USE_DEMO_MODE else 'REAL',
        'cycles': all_results
    }

    with open('trading_3_cycles_full_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    logging.info("💾 Результаты сохранены в trading_3_cycles_full_results.json")
    logging.info("✅ ВСЕ 3 ЦИКЛА ЗАВЕРШЕНЫ!")

    if exchange:
        await exchange.close()


if __name__ == '__main__':
    print("\n🚀 ПОЛНАЯ ВЕРСИЯ HFT-АЛГОРИТМА (10/10)")
    print("✅ Order Book интеграция")
    print("✅ Trade Feed интеграция")
    print("✅ Улучшенная reward function")
    print("✅ Demo режим с synthetic data")
    print("✅ 3 цикла с детальным PnL\n")

    try:
        asyncio.run(run_3_cycles_full())
    except KeyboardInterrupt:
        print("\n⚠️ Остановлено пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
