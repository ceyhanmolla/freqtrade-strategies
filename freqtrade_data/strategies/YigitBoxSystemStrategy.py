"""
YigitBoxSystemStrategy — Pivot + Williams %R Momentum

Mantik:
  Bir önceki günün H-L-C değerlerinden iki pivot seviyesi (PVT1, PVT2)
  hesaplanir. Williams %R(3) indikatoru ile momentum dogrulamasi yapilir.

  AL: Fiyat hem PVT1 hem de PVT2'nin üzerindeyken (bullish pivot kirilimi)
      WillR(3) > -20 ise (güçlü ivme) alim yapilir.

  SAT: Fiyat hem PVT1 hem de PVT2'nin altina düstügünde (bearish)
       WillR(3) < -70 ise (ivme tükendi) pozisyon kapatilir.

  Hesaplama (@informative('1d') ile bir önceki günün verisi):
    pvt1 = (H + L + C) / 3
    pvt2 = C
    willr_3 = Williams %R(3) periyot

  Backtest (88 gun, 24 pair, 100 USDT):
    Trade: 1142 | Kar: -$9.53 | Win%: 61.0% | DD: 22.6%
"""

from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
import talib.abstract as ta


class YigitBoxSystemStrategy(IStrategy):

    timeframe = '1h'
    startup_candle_count = 48

    minimal_roi = {
        "0": 0.15,
        "120": 0.05,
        "240": 0
    }
    stoploss = -0.05

    process_only_new_candles = True

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adam'] = dataframe['close'].shift(1)
        dataframe['tx'] = dataframe['high'].shift(1)
        dataframe['rx'] = dataframe['low'].shift(1)

        dataframe['pvt1'] = (dataframe['adam'] + dataframe['rx'] + dataframe['tx']) / 3
        dataframe['pvt2'] = dataframe['adam']

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['willr_3'] = ta.WILLR(dataframe, timeperiod=3)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['pvt2_1d']) &
                (dataframe['close'] > dataframe['pvt1_1d']) &
                (dataframe['willr_3'] > -20) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'ybs_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['pvt2_1d']) &
                (dataframe['close'] < dataframe['pvt1_1d']) &
                (dataframe['willr_3'] < -70) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'ybs_exit')

        return dataframe
