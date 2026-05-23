"""
YigitBoxStrategy — Pivot Box Breakout

Mantik:
  Bir önceki günün High, Low, Close değerlerinden 6 adet pivot seviyesi
  hesaplanir (r1, r2, s1, s2, pvt1, pvt2). Bu seviyeler gün boyunca
  sabit kalir ve bir "kutu" olarak düşünülebilir.

  AL: Fiyat r2 (üst kutuyu) yukari kestiginde break-out olur. 
      Yeni günün direnci asilir, yukselis beklenir.

  SAT: Fiyat s2 (alt kutuyu) asagi kestiginde break-down olur.
      Destek kirilir, pozisyon kapatilir.

  Hesaplama (@informative('1d') ile bir önceki günün verisi):
    pvt1 = (H + L + C) / 3
    r1 = 2*pvt1 - L ,  s1 = 2*pvt1 - H
    pvt2 = C
    r2 = pvt2 + (H-L)/2 ,  s2 = pvt2 - (H-L)/2

  Backtest (88 gun, 24 pair, 100 USDT):
    Trade: 604 | Kar: -$1.94 | Win%: 63.9% | DD: 15.1%
  
  Not: ALSAT filtre eklendiginde +$3.92'ye yukselir.
"""

from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib


class YigitBoxStrategy(IStrategy):

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
        dataframe['r1'] = (2 * dataframe['pvt1']) - dataframe['rx']
        dataframe['s1'] = (2 * dataframe['pvt1']) - dataframe['tx']

        dataframe['pvt2'] = dataframe['adam']
        dataframe['r2'] = dataframe['pvt2'] + ((dataframe['tx'] - dataframe['rx']) / 2)
        dataframe['s2'] = dataframe['pvt2'] - ((dataframe['tx'] - dataframe['rx']) / 2)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['box_upper'] = dataframe['r2_1d']
        dataframe['box_lower'] = dataframe['s2_1d']
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['box_upper']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'yigit_box_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['close'], dataframe['box_lower']) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'yigit_box_exit')

        return dataframe
