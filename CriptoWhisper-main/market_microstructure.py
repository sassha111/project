"""
Модуль рыночной микроструктуры для HFT-алгоритма
Интеграция Order Book и Trade Feed признаков
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple


class OrderBookAnalyzer:
    """
    Анализатор стакана ордеров (Order Book)
    Извлекает признаки для HFT-торговли
    """

    def __init__(self, levels: int = 10):
        """
        Args:
            levels: Количество уровней в стакане для анализа
        """
        self.levels = levels
        self.history = deque(maxlen=100)

    def calculate_features(self, orderbook: Dict) -> Dict[str, float]:
        """
        Рассчитать признаки из Order Book

        Args:
            orderbook: Dict с 'bids' и 'asks' [[price, volume], ...]

        Returns:
            Dict с признаками Order Book
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return self._get_default_features()

        bids = orderbook['bids'][:self.levels]
        asks = orderbook['asks'][:self.levels]

        if not bids or not asks:
            return self._get_default_features()

        features = {}

        # 1. Bid-Ask Spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        features['spread'] = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.001
        features['spread_bps'] = features['spread'] * 10000  # basis points

        # 2. Mid Price
        features['mid_price'] = (best_bid + best_ask) / 2

        # 3. Weighted Mid Price
        bid_volume = bids[0][1] if len(bids[0]) > 1 else 0
        ask_volume = asks[0][1] if len(asks[0]) > 1 else 0
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            features['weighted_mid_price'] = (best_bid * ask_volume + best_ask * bid_volume) / total_volume
        else:
            features['weighted_mid_price'] = features['mid_price']

        # 4. Depth Imbalance (дисбаланс глубины)
        total_bid_volume = sum([level[1] for level in bids if len(level) > 1])
        total_ask_volume = sum([level[1] for level in asks if len(level) > 1])
        total_depth = total_bid_volume + total_ask_volume

        if total_depth > 0:
            features['depth_imbalance'] = (total_bid_volume - total_ask_volume) / total_depth
        else:
            features['depth_imbalance'] = 0.0

        # 5. Volume at Best (объем на лучших уровнях)
        features['bid_volume_at_best'] = bid_volume
        features['ask_volume_at_best'] = ask_volume
        features['volume_imbalance_at_best'] = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0

        # 6. Order Book Pressure (давление на уровнях)
        bid_pressure = 0
        ask_pressure = 0

        for i, (price, volume) in enumerate(bids[:5]):
            if len(bids[0]) > 1:
                distance = abs(price - best_bid) if price != best_bid else 0.0001
                weight = 1 / (distance + 0.0001)  # Ближе = больший вес
                bid_pressure += volume * weight

        for i, (price, volume) in enumerate(asks[:5]):
            if len(asks[0]) > 1:
                distance = abs(price - best_ask) if price != best_ask else 0.0001
                weight = 1 / (distance + 0.0001)
                ask_pressure += volume * weight

        features['bid_pressure'] = bid_pressure
        features['ask_pressure'] = ask_pressure
        features['pressure_ratio'] = bid_pressure / ask_pressure if ask_pressure > 0 else 1.0

        # 7. Large Orders Detection (крупные заявки)
        bid_volumes = [level[1] for level in bids if len(level) > 1]
        ask_volumes = [level[1] for level in asks if len(level) > 1]

        avg_bid_size = np.mean(bid_volumes) if bid_volumes else 0
        avg_ask_size = np.mean(ask_volumes) if ask_volumes else 0

        large_bid_count = sum(1 for v in bid_volumes if v > 2 * avg_bid_size)
        large_ask_count = sum(1 for v in ask_volumes if v > 2 * avg_ask_size)
        total_large = large_bid_count + large_ask_count

        if total_large > 0:
            features['large_order_imbalance'] = (large_bid_count - large_ask_count) / total_large
        else:
            features['large_order_imbalance'] = 0.0

        features['large_bids'] = large_bid_count
        features['large_asks'] = large_ask_count

        # 8. Depth at levels
        features['total_bid_volume'] = total_bid_volume
        features['total_ask_volume'] = total_ask_volume

        # Сохранить в историю
        self.history.append(features)

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Признаки по умолчанию при отсутствии данных"""
        return {
            'spread': 0.001,
            'spread_bps': 10.0,
            'mid_price': 0.0,
            'weighted_mid_price': 0.0,
            'depth_imbalance': 0.0,
            'bid_volume_at_best': 0.0,
            'ask_volume_at_best': 0.0,
            'volume_imbalance_at_best': 0.0,
            'bid_pressure': 0.0,
            'ask_pressure': 0.0,
            'pressure_ratio': 1.0,
            'large_order_imbalance': 0.0,
            'large_bids': 0,
            'large_asks': 0,
            'total_bid_volume': 0.0,
            'total_ask_volume': 0.0
        }

    def get_trend(self, window: int = 20) -> Dict[str, float]:
        """
        Получить тренд признаков за последние N обновлений

        Args:
            window: Размер окна для анализа тренда

        Returns:
            Dict с трендами признаков
        """
        if len(self.history) < 2:
            return {}

        recent = list(self.history)[-window:]
        trends = {}

        for key in recent[0].keys():
            if key not in ['large_bids', 'large_asks']:  # Пропустить счетчики
                values = [h[key] for h in recent if key in h]
                if len(values) >= 2:
                    trend = values[-1] - values[0]
                    trends[f'{key}_trend'] = trend

        return trends


class TradeFlowAnalyzer:
    """
    Анализатор потока сделок (Trade Feed)
    Извлекает признаки агрессивности и направления
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Размер окна для анализа сделок
        """
        self.window_size = window_size
        self.trades_history = deque(maxlen=window_size)

    def calculate_features(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Рассчитать признаки из Trade Feed

        Args:
            trades: List of trades [{'price': float, 'amount': float, 'side': 'buy'/'sell', 'timestamp': int}, ...]

        Returns:
            Dict с признаками Trade Feed
        """
        if not trades or len(trades) == 0:
            return self._get_default_features()

        # Добавить в историю
        self.trades_history.extend(trades)

        features = {}

        # Использовать последние N сделок
        recent_trades = list(self.trades_history)[-self.window_size:]

        if not recent_trades:
            return self._get_default_features()

        # 1. Buy/Sell Volume Ratio
        buy_volume = sum([t['amount'] for t in recent_trades if t.get('side') == 'buy'])
        sell_volume = sum([t['amount'] for t in recent_trades if t.get('side') == 'sell'])
        total_volume = buy_volume + sell_volume

        if sell_volume > 0:
            features['buy_sell_ratio'] = buy_volume / sell_volume
        else:
            features['buy_sell_ratio'] = 2.0 if buy_volume > 0 else 1.0

        if total_volume > 0:
            features['buy_volume_pct'] = buy_volume / total_volume
            features['sell_volume_pct'] = sell_volume / total_volume
        else:
            features['buy_volume_pct'] = 0.5
            features['sell_volume_pct'] = 0.5

        # 2. Trade Aggressiveness (buy vs sell initiated)
        buy_count = sum([1 for t in recent_trades if t.get('side') == 'buy'])
        sell_count = sum([1 for t in recent_trades if t.get('side') == 'sell'])
        total_count = len(recent_trades)

        if total_count > 0:
            features['trade_aggressiveness'] = (buy_count - sell_count) / total_count
        else:
            features['trade_aggressiveness'] = 0.0

        # 3. VWAP (Volume Weighted Average Price)
        total_value = sum([t['price'] * t['amount'] for t in recent_trades])
        if total_volume > 0:
            features['vwap'] = total_value / total_volume
            current_price = recent_trades[-1]['price']
            features['vwap_distance'] = (current_price - features['vwap']) / features['vwap']
        else:
            features['vwap'] = recent_trades[-1]['price'] if recent_trades else 0
            features['vwap_distance'] = 0.0

        # 4. Trade Size Distribution
        trade_sizes = [t['amount'] for t in recent_trades]
        features['avg_trade_size'] = np.mean(trade_sizes)
        features['std_trade_size'] = np.std(trade_sizes) if len(trade_sizes) > 1 else 0

        if features['std_trade_size'] > 0:
            threshold = features['avg_trade_size'] + 2 * features['std_trade_size']
            large_trades = [t for t in recent_trades if t['amount'] > threshold]
            features['large_trades_ratio'] = len(large_trades) / len(recent_trades)

            # Разделить на buy/sell
            large_buys = sum([1 for t in large_trades if t.get('side') == 'buy'])
            large_sells = sum([1 for t in large_trades if t.get('side') == 'sell'])
            total_large = len(large_trades)
            if total_large > 0:
                features['large_trade_imbalance'] = (large_buys - large_sells) / total_large
            else:
                features['large_trade_imbalance'] = 0.0
        else:
            features['large_trades_ratio'] = 0.0
            features['large_trade_imbalance'] = 0.0

        # 5. Trade Momentum (with time decay)
        if len(recent_trades) > 1:
            current_time = recent_trades[-1].get('timestamp', 0)
            momentum = 0

            for trade in recent_trades:
                volume = trade['amount']
                side = 1 if trade.get('side') == 'buy' else -1
                time_diff = current_time - trade.get('timestamp', current_time)
                time_weight = 1 / (time_diff / 1000 + 1)  # Decay by time (ms to s)

                momentum += side * volume * time_weight

            features['trade_momentum'] = momentum / len(recent_trades)
        else:
            features['trade_momentum'] = 0.0

        # 6. Trade Frequency
        if len(recent_trades) > 1:
            time_span = recent_trades[-1].get('timestamp', 0) - recent_trades[0].get('timestamp', 0)
            if time_span > 0:
                features['trade_frequency'] = len(recent_trades) / (time_span / 1000)  # trades per second
            else:
                features['trade_frequency'] = 0.0
        else:
            features['trade_frequency'] = 0.0

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Признаки по умолчанию при отсутствии данных"""
        return {
            'buy_sell_ratio': 1.0,
            'buy_volume_pct': 0.5,
            'sell_volume_pct': 0.5,
            'trade_aggressiveness': 0.0,
            'vwap': 0.0,
            'vwap_distance': 0.0,
            'avg_trade_size': 0.0,
            'std_trade_size': 0.0,
            'large_trades_ratio': 0.0,
            'large_trade_imbalance': 0.0,
            'trade_momentum': 0.0,
            'trade_frequency': 0.0
        }


class MarketMicrostructureFeatures:
    """
    Объединенный класс для всех признаков рыночной микроструктуры
    """

    def __init__(self):
        self.orderbook_analyzer = OrderBookAnalyzer(levels=10)
        self.tradeflow_analyzer = TradeFlowAnalyzer(window_size=100)

    def extract_features(self, orderbook: Optional[Dict], trades: Optional[List[Dict]]) -> Dict[str, float]:
        """
        Извлечь все признаки микроструктуры

        Args:
            orderbook: Order book data
            trades: Trade feed data

        Returns:
            Dict со всеми признаками
        """
        features = {}

        # Order Book признаки
        if orderbook:
            ob_features = self.orderbook_analyzer.calculate_features(orderbook)
            features.update(ob_features)

        # Trade Feed признаки
        if trades:
            trade_features = self.tradeflow_analyzer.calculate_features(trades)
            features.update(trade_features)

        return features

    def get_feature_names(self) -> List[str]:
        """Получить список всех названий признаков"""
        ob_features = self.orderbook_analyzer._get_default_features()
        trade_features = self.tradeflow_analyzer._get_default_features()

        all_features = list(ob_features.keys()) + list(trade_features.keys())
        return all_features


def calculate_hft_reward_with_microstructure(
    profit: float,
    duration: float,
    microstructure_features: Dict[str, float],
    entry_features: Dict[str, float]
) -> float:
    """
    Улучшенная функция награды для HFT с учетом микроструктуры рынка

    Args:
        profit: Прибыль/убыток сделки
        duration: Длительность позиции (секунды)
        microstructure_features: Признаки микроструктуры на момент закрытия
        entry_features: Признаки микроструктуры на момент входа

    Returns:
        Итоговая награда
    """
    # Базовая награда за прибыль
    base_reward = profit

    # 1. Бонус за быстрое исполнение (HFT)
    if duration < 10:  # < 10 секунд
        speed_bonus = 0.5
    elif duration < 60:  # < 1 минуты
        speed_bonus = 0.2
    elif duration < 180:  # < 3 минут
        speed_bonus = 0.1
    else:
        speed_bonus = 0

    # 2. Штраф за широкий spread (плохая ликвидность)
    spread = microstructure_features.get('spread', 0.001)
    if spread > 0.002:  # > 0.2%
        spread_penalty = -0.3
    elif spread > 0.001:  # > 0.1%
        spread_penalty = -0.1
    else:
        spread_penalty = 0

    # 3. Бонус за торговлю в направлении depth imbalance
    depth_imbalance_entry = entry_features.get('depth_imbalance', 0)
    alignment_bonus = 0

    if profit > 0 and depth_imbalance_entry > 0.1:  # Прибыльная long при давлении покупателей
        alignment_bonus = 0.3
    elif profit > 0 and depth_imbalance_entry < -0.1:  # Прибыльная short при давлении продавцов
        alignment_bonus = 0.3
    elif profit < 0 and abs(depth_imbalance_entry) < 0.05:  # Убыточная при равновесии
        alignment_bonus = 0  # Не штрафуем, рынок был неопределенным

    # 4. Бонус за торговлю в направлении trade momentum
    trade_momentum_entry = entry_features.get('trade_momentum', 0)
    momentum_bonus = 0

    if profit > 0 and trade_momentum_entry > 0:
        momentum_bonus = 0.2
    elif profit > 0 and trade_momentum_entry < 0:
        momentum_bonus = 0.2

    # 5. Штраф за торговлю против агрессивности
    trade_aggressiveness = entry_features.get('trade_aggressiveness', 0)
    aggressiveness_penalty = 0

    if profit < 0 and abs(trade_aggressiveness) > 0.3:
        # Убыток при сильной агрессивности - плохой вход
        aggressiveness_penalty = -0.2

    # 6. Бонус за торговлю с крупными участниками
    large_order_imbalance = entry_features.get('large_order_imbalance', 0)
    large_player_bonus = 0

    if profit > 0 and abs(large_order_imbalance) > 0.3:
        # Прибыльная сделка в направлении китов
        large_player_bonus = 0.15

    # Итоговая награда
    total_reward = (
        base_reward +
        speed_bonus +
        spread_penalty +
        alignment_bonus +
        momentum_bonus +
        aggressiveness_penalty +
        large_player_bonus
    )

    return total_reward


# Вспомогательные функции для генерации синтетических данных (для demo режима)

def generate_synthetic_orderbook(current_price: float, volatility: float = 0.01) -> Dict:
    """
    Генерация синтетического Order Book для demo режима

    Args:
        current_price: Текущая цена
        volatility: Волатильность для генерации уровней

    Returns:
        Dict с 'bids' и 'asks'
    """
    np.random.seed()

    # Генерация bid уровней
    bids = []
    for i in range(10):
        price = current_price * (1 - volatility * (i + 1) * 0.1)
        volume = np.random.uniform(0.1, 5.0) * (1 - i * 0.05)  # Больше объем на лучших уровнях
        bids.append([price, volume])

    # Генерация ask уровней
    asks = []
    for i in range(10):
        price = current_price * (1 + volatility * (i + 1) * 0.1)
        volume = np.random.uniform(0.1, 5.0) * (1 - i * 0.05)
        asks.append([price, volume])

    return {'bids': bids, 'asks': asks, 'timestamp': pd.Timestamp.now().value // 10**6}


def generate_synthetic_trades(current_price: float, num_trades: int = 20) -> List[Dict]:
    """
    Генерация синтетического Trade Feed для demo режима

    Args:
        current_price: Текущая цена
        num_trades: Количество сделок для генерации

    Returns:
        List of synthetic trades
    """
    np.random.seed()

    trades = []
    timestamp = pd.Timestamp.now().value // 10**6

    for i in range(num_trades):
        side = 'buy' if np.random.random() > 0.5 else 'sell'
        price = current_price * (1 + np.random.normal(0, 0.0005))
        amount = np.random.uniform(0.01, 1.0)

        trades.append({
            'price': price,
            'amount': amount,
            'side': side,
            'timestamp': timestamp - (num_trades - i) * 1000  # Распределить по времени
        })

    return trades
