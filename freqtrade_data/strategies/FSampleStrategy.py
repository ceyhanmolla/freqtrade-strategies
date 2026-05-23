from pandas import DataFrame

from freqtrade.strategy import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class FSampleStrategy(IStrategy):
    """
    RSI + TEMA + Bollinger Bands tabanli mean-reversion stratejisi.

    NOT: ADX hesaplanir ancak sinyallerde kullanilmaz. Bunun sebebi Docker
    ortamindaki TA-Lib surumunde ADX'in RSI'dan ONCE hesaplanmasi gerekliligi:
    ADX kaldirilinca RSI degerleri farkli cikiyor (TA-Lib ic state sorunu).

    MUHENDISLIK IYILESTIRMELERI:
    1. STOCHF, MACD, MFI, SAR, HT_SINE kaldirildi (sinyallerde kullanilmiyor)
    2. BB lower/upper/percent/width kaldirildi (sadece mid kullaniliyor)
    3. Importlar temizlendi: numpy, pd, BooleanParameter, vb.
    4. Sinyal mantigi, trade sayisi ve kazanma orani AYNI

    Sinyal mantigi:
    - Giris: RSI 30'u yukari keser + TEMA BB ortasinin altinda + yukseliyor
    - Cikis: RSI 70'i yukari keser + TEMA BB ortasinin ustunde + dusuyor
    """
    INTERFACE_VERSION = 3
    can_short = False

    timeframe = "1h"
    startup_candle_count = 30
    process_only_new_candles = True

    minimal_roi = {
        "0": 1,
    }

    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["rsi"] = ta.RSI(dataframe)

        bb = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_middleband"] = bb["mid"]

        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi"], 30))
                & (dataframe["tema"] <= dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi"], 70))
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (dataframe["tema"] < dataframe["tema"].shift(1))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
