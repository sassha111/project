#!/usr/bin/env python3
"""
Script to run the high-frequency trading algorithm with 1% risk setting
Supports trading on 10 liquid pairs simultaneously
Features fast-start training - begins training after collecting 100 ticks
"""

import asyncio
import logging
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Main import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

if __name__ == "__main__":
    print("Запуск алгоритма высокочастотной торговли с риском 1%")
    print("Поддерживаемые пары: DOGE/USDT:USDT, BTC/USDT:USDT, ETH/USDT:USDT, BNB/USDT:USDT, SOL/USDT:USDT, XRP/USDT:USDT, ADA/USDT:USDT, AVAX/USDT:USDT, DOT/USDT:USDT, MATIC/USDT:USDT")
    print("Алгоритм начнет обучение после сбора 100 тиков (примерно 1-2 минуты)")
    print("Для остановки нажмите Ctrl+C")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nАлгоритм остановлен пользователем")
    except Exception as e:
        print(f"Ошибка при запуске алгоритма: {e}")
        logging.error(f"Ошибка при запуске алгоритма: {e}")