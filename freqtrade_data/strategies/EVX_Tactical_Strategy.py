# --- EVX Tactical Strategy ---

import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy

class EVX_Tactical_Strategy(IStrategy):
    """
    EVX (Excess Volume Index) Tabanli Taktiksel Hacim Stratejisi

    MANTIK:
    - Her mumda bid/ask hacmini (close-low)/(high-low) oraniyla tahmin eder
    - B = volume * buy_pressure (alici hacmi)
    - A = volume - B (satici hacmi)
    - EVX = (B - A) / A (asiri hacim endeksi)
    - det_buy / det_sell = matris determinantlari (fiyat-hacim iliskisi)

    GIRIS: EVX > 0 (alicilar baskin)
    CIKIS: det_sell < 0 VE EVX < -0.02 (saticilar baskin + fiyat dusus sinyali)

    BACKTEST (30 gun, 15m, 2026-04-03 → 2026-05-04):
    - Profit: +12.66%
    - Trades: 21
    - Win Rate: 66.7%
    - Drawdown: 0.14%
    - Sharpe: 3.63 | Calmar: 122.70

    GUCLU YONLER: Cok dusuk drawdown, matematiksel temel, tum TF'lerde karli
    ZAYIF YONLER: Tek indikator, trend korelasyonu yok, sadece LONG
    """
    INTERFACE_VERSION = 3

    timeframe = '15m'

    startup_candle_count = 50
    process_only_new_candles = True

    use_exit_signal = True

    minimal_roi = {
        "0": 0.15,
        "120": 0.05,
        "240": 0.02,
        "480": 0
    }

    stoploss = -0.074

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        high_low_diff = dataframe['high'] - dataframe['low']
        high_low_diff = high_low_diff.replace(0, np.nan)

        buy_pressure = (dataframe['close'] - dataframe['low']) / high_low_diff
        buy_pressure = buy_pressure.fillna(0.5)

        dataframe['B'] = dataframe['volume'] * buy_pressure
        dataframe['A'] = dataframe['volume'] - dataframe['B']

        safe_A = dataframe['A'].replace(0, np.nan)
        dataframe['EVX'] = (dataframe['B'] - safe_A) / safe_A
        dataframe['EVX'] = dataframe['EVX'].fillna(0)

        dataframe['det_buy'] = (dataframe['A'] * dataframe['open']) - (dataframe['B'] * dataframe['close'])
        dataframe['det_sell'] = (dataframe['A'] * dataframe['close']) - (dataframe['B'] * dataframe['open'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe.loc[
            (
                (dataframe['EVX'] > 0) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe.loc[
            (
                (dataframe['det_sell'] < 0) &
                (dataframe['EVX'] < -0.02) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
