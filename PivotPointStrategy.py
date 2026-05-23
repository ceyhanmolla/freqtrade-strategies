"""
PivotPointStrategy — Camarilla Pivot Mean Reversion

Mantik:
  Camarilla pivot formulü, bir önceki periyodun H-L-C değerlerini kullanarak
  7 adet destek (S1-S4) ve direnç (R1-R4) seviyesi hesaplar.

  Bu strateji, Camarilla'nın mean reversion (ortalamaya dönüş) mantığıyla
  çalışır:

  AL: Fiyat S3-S4 bölgesine (7 günlük aşırı satım) düştüğünde işleme girer.
  Bu seviyeler fiyatın istatistiksel olarak en dip bölgesidir.

  SAT: Fiyat R1 veya R2 seviyesine (ilk direnç) döndüğünde kâr alır.
  Dibe yakın alıp, ilk tepede satarak mean reversion'dan faydalanır.

  Hesaplama:
    window = 168 (7 gun x 24 saat)
    h = son 7 günün en yükseği
    l = son 7 günün en düşüğü  
    c = bir önceki mumun kapanışı
    r = h - l (7 günlük range)

    S3 = c - r * 1.1 / 4
    S4 = c - r * 1.1 / 2  
    R1 = c + r * 1.1 / 12
    R2 = c + r * 1.1 / 6

  Backtest (88 gun, 24 pair, 100 USDT):
    Trade: 46 | Kar: +$0.28 | Win%: 71.7% | DD: 2.56%

  Not: Her pair kendi 1h verisinden hesaplama yapar, 
  BTC'ye veya harici veriye bağımlı değildir.
"""

from pandas import DataFrame
from freqtrade.strategy import IStrategy


class PivotPointStrategy(IStrategy):

    timeframe = '1h'
    startup_candle_count = 200

    minimal_roi = {
        "0": 0.15,
        "120": 0.05,
        "240": 0
    }
    stoploss = -0.05

    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        window = 168
        h = dataframe['high'].rolling(window=window).max().shift(1)
        l = dataframe['low'].rolling(window=window).min().shift(1)
        c = dataframe['close'].shift(1)
        r = h - l

        dataframe['s3'] = c - r * 1.1 / 4
        dataframe['s4'] = c - r * 1.1 / 2
        dataframe['r1'] = c + r * 1.1 / 12
        dataframe['r2'] = c + r * 1.1 / 6

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] <= dataframe['s3']) &
                (dataframe['close'] >= dataframe['s4']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'cam_buy_s3')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['r1']) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'cam_sell_r1')

        dataframe.loc[
            (
                (dataframe['close'] >= dataframe['r2']) &
                (dataframe['volume'] > 0)
            ),
            ['exit_long', 'exit_tag']] = (1, 'cam_sell_r2')

        return dataframe
