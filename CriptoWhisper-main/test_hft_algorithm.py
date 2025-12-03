#!/usr/bin/env python3
"""
Тестовый скрипт для демонстрации работы HFT-алгоритма
Использует синтетические данные для симуляции торговли
"""

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from Main import TradingEnvironment, get_or_train_model_sync, backtest_model_sync

def generate_synthetic_crypto_data(num_ticks=1000, initial_price=50000, volatility=0.02):
    """
    Генерирует синтетические крипто-данные для тестирования
    Имитирует поведение BTC/USDT с реалистичными движениями цены
    """
    logging.info(f"🔧 Генерация {num_ticks} синтетических тиков (волатильность={volatility*100}%)")

    timestamps = [datetime.now() - timedelta(seconds=num_ticks - i) for i in range(num_ticks)]

    # Generate realistic price movements using geometric Brownian motion
    returns = np.random.normal(0, volatility, num_ticks)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = []
    for i in range(num_ticks):
        price = prices[i]
        spread = price * 0.0002  # 0.02% spread

        open_price = price + np.random.uniform(-spread, spread)
        high_price = max(open_price, price + abs(np.random.normal(0, spread)))
        low_price = min(open_price, price - abs(np.random.normal(0, spread)))
        close_price = price
        volume = np.random.uniform(100, 1000)

        # Calculate ATR-like volatility
        true_range = high_price - low_price
        atr = true_range * (1 + np.random.uniform(-0.2, 0.2))

        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'atr': atr,
            'price_change': (close_price - open_price) / open_price if i > 0 else 0,
            'spread': spread,
            'bid_price': close_price - spread/2,
            'ask_price': close_price + spread/2,
            'bid_volume': volume * np.random.uniform(0.4, 0.6),
            'ask_volume': volume * np.random.uniform(0.4, 0.6),
        })

    df = pd.DataFrame(data)
    logging.info(f"✅ Сгенерировано {len(df)} тиков | Цена: ${prices[0]:.2f} → ${prices[-1]:.2f} | Изменение: {((prices[-1]/prices[0])-1)*100:.2f}%")
    return df

def test_trading_environment(df):
    """Тестирование торговой среды"""
    logging.info("\n" + "="*80)
    logging.info("🏗️  ТЕСТ 1: Проверка торговой среды (TradingEnvironment)")
    logging.info("="*80)

    # Test basic environment functionality
    env = TradingEnvironment(
        data=df,
        initial_balance=10000,
        risk_percentage=0.01,
        max_position_duration=180  # HFT: max 3 minutes
    )

    logging.info(f"✅ Среда инициализирована | Баланс: ${env.initial_balance} | Риск: {env.risk_percentage*100}%")
    logging.info(f"   • Max позиция: {env.max_position_duration} сек | SL: {env.sl_multiplier}x ATR | TP: {env.tp_multiplier}x ATR")

    # Test a few random actions
    obs, info = env.reset()
    logging.info(f"✅ Среда сброшена | Observation shape: {obs.shape}")

    total_reward = 0
    actions_taken = {'hold': 0, 'long': 0, 'short': 0}

    # Simulate 50 steps
    for step in range(50):
        action = np.random.choice([0, 1, 2])  # Random action
        action_names = ['hold', 'long', 'short']
        actions_taken[action_names[action]] += 1

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if done or truncated:
            logging.info(f"   • Episode завершен на шаге {step+1}")
            obs, info = env.reset()
            break

    logging.info(f"✅ Тест среды завершен | Total reward: {total_reward:.4f}")
    logging.info(f"   • Действия: Hold={actions_taken['hold']}, Long={actions_taken['long']}, Short={actions_taken['short']}")

    # Test Kelly Criterion Risk Manager
    logging.info(f"\n📊 Kelly Criterion Statistics:")
    stats = env.risk_manager.get_statistics()
    logging.info(f"   • Win Rate: {stats['win_rate']*100:.1f}% | Profit Factor: {stats['profit_factor']:.2f}")
    logging.info(f"   • Avg Win: ${stats['avg_win']:.2f} | Avg Loss: ${stats['avg_loss']:.2f}")
    if 'sharpe_ratio' in stats and 'max_drawdown' in stats:
        logging.info(f"   • Sharpe Ratio: {stats['sharpe_ratio']:.2f} | Max DD: {stats['max_drawdown']*100:.2f}%")

def test_model_training(train_df, test_df):
    """Тестирование обучения модели PPO"""
    logging.info("\n" + "="*80)
    logging.info("🎓 ТЕСТ 2: Обучение модели PPO (500K шагов)")
    logging.info("="*80)

    symbol = "BTC_USDT_TEST"
    models_dir = "test_models"

    logging.info(f"🔧 Параметры обучения:")
    logging.info(f"   • Train data: {len(train_df)} тиков | Test data: {len(test_df)} тиков")
    logging.info(f"   • Timesteps: 500,000 (HFT улучшение, было 50K)")
    logging.info(f"   • Risk Management: Kelly Criterion (0.5-2%)")
    logging.info(f"   • Reward Function: Multi-component HFT reward")

    logging.info("\n⏳ Начало обучения... (это займет ~10-15 минут)")

    # Train model
    model, norm_params = get_or_train_model_sync(
        symbol=symbol,
        train_df=train_df,
        models_dir=models_dir,
        force_train=True
    )

    logging.info(f"✅ Модель обучена успешно!")
    logging.info(f"   • Norm params: mean_price={norm_params['mean_price']:.2f}, std_price={norm_params['std_price']:.2f}")

    return model, norm_params

def test_backtesting(model, test_df, symbol, norm_params):
    """Тестирование backtesting"""
    logging.info("\n" + "="*80)
    logging.info("📈 ТЕСТ 3: Backtesting на тестовых данных")
    logging.info("="*80)

    results = backtest_model_sync(model, test_df, symbol, norm_params)

    logging.info(f"\n🎯 РЕЗУЛЬТАТЫ BACKTESTING:")
    logging.info(f"   • Total Return: {results['total_return']*100:.2f}%")
    logging.info(f"   • Total Trades: {results['total_trades']}")
    logging.info(f"   • Win Rate: {results['win_rate']*100:.1f}%")
    logging.info(f"   • Profit Factor: {results['profit_factor']:.2f}")
    logging.info(f"   • Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logging.info(f"   • Max Drawdown: {results['max_drawdown']*100:.2f}%")
    logging.info(f"   • Final Balance: ${results['final_balance']:.2f}")

    if 'profitable_trades' in results:
        logging.info(f"   • Profitable Trades: {results['profitable_trades']}")
        logging.info(f"   • Losing Trades: {results['losing_trades']}")

    return results

def main():
    """Главная функция тестирования"""
    logging.info("\n" + "="*100)
    logging.info("🚀 ТЕСТИРОВАНИЕ HFT-АЛГОРИТМА CRIPTOWHISPER С УЛУЧШЕНИЯМИ")
    logging.info("="*100)
    logging.info("\n✅ Внедренные улучшения:")
    logging.info("   1. ✅ Stop-Loss/Take-Profit для HFT (1.5x/2.5x ATR)")
    logging.info("   2. ✅ Максимальное время позиции (180 секунд)")
    logging.info("   3. ✅ Продвинутая reward function (speed bonus, duration penalty)")
    logging.info("   4. ✅ Kelly Criterion для адаптивного sizing (0.5-2%)")
    logging.info("   5. ✅ Увеличенные timesteps (500K initial, 100K retrain)")
    logging.info("   6. ✅ Оптимизация гиперпараметров (50 trials, было 3)")
    logging.info("\n" + "="*100 + "\n")

    # Generate synthetic data
    num_ticks = 1000
    df = generate_synthetic_crypto_data(num_ticks=num_ticks, initial_price=50000, volatility=0.02)

    # Split into train and test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    logging.info(f"\n📊 Разделение данных: Train={len(train_df)} тиков | Test={len(test_df)} тиков\n")

    # Run tests
    test_trading_environment(df)

    model, norm_params = test_model_training(train_df, test_df)

    results = test_backtesting(model, test_df, "BTC_USDT_TEST", norm_params)

    # Final summary
    logging.info("\n" + "="*100)
    logging.info("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    logging.info("="*100)
    logging.info("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    logging.info(f"   • Модель обучена на {len(train_df)} тиках")
    logging.info(f"   • Backtesting на {len(test_df)} тиках")
    logging.info(f"   • Return: {results['total_return']*100:.2f}% | Win Rate: {results['win_rate']*100:.1f}%")
    logging.info(f"   • Sharpe: {results['sharpe_ratio']:.2f} | Max DD: {results['max_drawdown']*100:.2f}%")
    logging.info(f"   • Total Trades: {results['total_trades']} | Profit Factor: {results['profit_factor']:.2f}")
    logging.info("\n✅ HFT-алгоритм готов к работе с реальными данными!")
    logging.info("="*100 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n❌ Тестирование прервано пользователем")
    except Exception as e:
        logging.error(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
