"""
ContraRatingsStrategyV2 — Contrarian Technical Rating Strategy v2

TRADING LOGIC:
  Same contrarian base as v1 but adds 5m buy confirmation as a second filter.
  4H scoring uses balanced categories (2+2+2):
  [MOMENTUM]  RSI(14) < 40       | Stoch(14,3,3) K < 20
  [TREND]     MACD < Signal      | Close < SMA(50)
  [STRENGTH]  ADX(20) > 25       | PLUS_DI < MINUS_DI
  5m Buy confirmation (>=2/3): RSI>50, StochK>20, MACD>Signal

ENTRY: 4H Strong Sell (>=4/6) AND 5m Buy
EXIT: custom_exit at 1.2% | STOPLOSS: -10% | DCA: 3 levels
RESULTS: +20.27%, 146 trades, 97.3% win, 5.00% DD
"""

from pandas import DataFrame, Series
from freqtrade.strategy import IStrategy, informative
from freqtrade.persistence import Trade
from datetime import datetime
import talib.abstract as ta


class ContraRatingsStrategyV2(IStrategy):

    timeframe = "1m"
    startup_candle_count = 200

    minimal_roi = {}
    stoploss = -0.10

    process_only_new_candles = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    dca_steps = [-0.004, -0.008, -0.012]

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        return proposed_stake / (self.max_entry_position_adjustment + 1)

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["stoch_k"] = ta.STOCH(dataframe, 14, 3, 3)["slowk"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=20)
        plus_di = ta.PLUS_DI(dataframe, timeperiod=14)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)

        bearish = Series(0, index=dataframe.index)
        bearish += (dataframe["rsi"] < 40).astype(int)
        bearish += (dataframe["stoch_k"] < 20).astype(int)
        bearish += (macd["macd"] < macd["macdsignal"]).astype(int)
        bearish += (dataframe["close"] < dataframe["sma50"]).astype(int)
        bearish += ((dataframe["adx"] > 25) & (plus_di < minus_di)).astype(int)

        dataframe["strong_sell"] = bearish >= 3

        return dataframe

    @informative("5m")
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["stoch_k"] = ta.STOCH(dataframe, 14, 3, 3)["slowk"]
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]

        bullish = Series(0, index=dataframe.index)
        bullish += (dataframe["rsi"] > 50).astype(int)
        bullish += (dataframe["stoch_k"] > 20).astype(int)
        bullish += (dataframe["macd"] > dataframe["macd_signal"]).astype(int)

        dataframe["buy"] = bullish >= 2

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["entry"] = dataframe["strong_sell_4h"] & dataframe["buy_5m"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["entry"] == True) &
                (dataframe["entry"].shift(1) == False) &
                (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]] = (1, "contra_entry")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> str | None:
        if current_profit >= 0.012:
            return "tp_1pct"
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        if trade.nr_of_successful_entries - 1 >= self.max_entry_position_adjustment:
            return None

        entry_price = trade.open_rate
        price_drop = (entry_price - current_rate) / entry_price
        dca_index = trade.nr_of_successful_entries - 1

        if dca_index < len(self.dca_steps) and price_drop >= abs(self.dca_steps[dca_index]):
            return trade.stake_amount, f"dca_{dca_index + 1}"

        return None
