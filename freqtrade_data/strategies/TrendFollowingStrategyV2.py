from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
from datetime import datetime
from freqtrade.persistence import Trade

import talib.abstract as ta


class TrendFollowingStrategyV2(IStrategy):
    """
    V1'in basarili mantigi + 1h EMA(50) filtre.

    V1 basarisi (+%17.66):
    - Giris: close EMA(20)'yi yukari keser + OBV yukseliyor
    - Cikis: close EMA(20)'yi asagi keser + OBV yukseliyor (bilincli tasarim:
      OBV hala yukseliyorsa sadece gecici bir geri cekilme, trend devam eder)
    - trailing 0.05/0.10

    V2 sorunu (-%30.71):
    - use_exit_signal=False (cikis yok)
    - leverage 1.5x (spotta gereksiz)
    - ADX esigi 20 (cok dusuk, cok fazla sinyal)
    - trailing 0.02/0.04 (cok siki, erken cikis)

    Cozum: V1'in exit mantigini + 1h EMA filtresini + V1'in trailing'ini kullan.
    """
    INTERFACE_VERSION = 3
    can_short = False

    timeframe = '5m'
    startup_candle_count = 200
    process_only_new_candles = True

    minimal_roi = {"0": 0.15, "30": 0.10, "60": 0.05}

    stoploss = -0.15

    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    use_exit_signal = True

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:
        return -0.15

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['trend']) &
                (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
                (dataframe['obv'] > dataframe['obv'].shift(1)) &
                (dataframe['close_1h'] > dataframe['ema_50_1h']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'trend_long_1h')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['trend']) &
                (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
                (dataframe['obv'] > dataframe['obv'].shift(1))
            ),
            'exit_long',
        ] = 1
        return dataframe
