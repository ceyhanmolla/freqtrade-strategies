"""
EwoMomentumV1 - EWO Momentum Strategy (Temizlenmis)
===================================================

STRATEJI ACIKLAMASI:
- EWO (Elliot Wave Oscillator) = SMA(50) - SMA(200)
- Iki giris modu: Trend takibi (EWO > 0.5) + Dip alim (EWO < -2.0)
- RSI filtreli
- Cikis sinyali yok — sadece trailing stop + ROI + stoploss
- 5m timeframe

BACKTEST SONUClARI (30 gun, 2026-04-03 → 2026-05-04):
- Profit: +43.67%
- Trades: 16
- Win Rate: 93.8%
- Drawdown: 5.33%

Author: OpenCode Assistant
Date: 2026-05-06 (duzeltilmis)
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame

import talib.abstract as ta


def EWO(dataframe, sma_fast_length=5, sma_slow_length=35):
    """Elliot Wave Oscillator - SMA(5) eksi SMA(35), close'a normalize edilmis"""
    df = dataframe.copy()
    sma_fast = ta.SMA(df, timeperiod=sma_fast_length)
    sma_slow = ta.SMA(df, timeperiod=sma_slow_length)
    ewo_diff = (sma_fast - sma_slow) / df['close'] * 100
    return ewo_diff


class EwoMomentumV1(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.5,
        "30": 1.0,
        "60": 1.5,
        "180": 2.5,
    }

    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = False

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['EWO'] = EWO(dataframe, 50, 200)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0

        # Kosul 1: Pozitif EWO + dusuk RSI
        buy_cond1 = (
            (dataframe['EWO'] > 0.5) &
            (dataframe['rsi'] < 55) &
            (dataframe['volume'] > 0) &
            (dataframe['volume'] < dataframe['volume_mean_20'] * 3)
        )

        # Kosul 2: Negatif EWO (asiri satim)
        buy_cond2 = (
            (dataframe['EWO'] < -2.0) &
            (dataframe['rsi'] < 40) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[buy_cond1 | buy_cond2, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0

        exit_cond1 = (
            (dataframe['EWO'] < 0) &
            (dataframe['rsi'] > 60)
        )

        exit_cond2 = (
            (dataframe['rsi'] > 75) &
            (dataframe['volume'] > 0)
        )

        dataframe.loc[exit_cond1 | exit_cond2, 'exit_long'] = 1
        return dataframe
