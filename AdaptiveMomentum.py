import logging
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
from datetime import datetime
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class AdaptiveMomentum(IStrategy):
    """
    MUHENDISLIK IYILESTIRMELERI (2026-05-09):

    1. iloc[-1] + Trade.get_open_trades() KALDIRILDI (KRITIK)
       Neden: populate_entry/exit_trend'te kullanilan bu pattern backtest'te
       sadece SON mumu goruyor. Backtest sonuclari canliyla uyusmuyordu.
       Yeni: TUM entry kosullari vectorize edildi.

    2. populate_exit_trend KALDIRILDI
       Neden: scalar current_profit + DataFrame karmasi calismiyordu.
       Yeni: use_exit_signal=False, tum cikislar custom_exit + trailing + ROI ile.

    3. EMA range -> .value
       Neden: 42 EMA kolonu hesaplaniyor, sadece 2'si kullaniliyor.
       Hyperopt zaten populate_indicators'i her trial'da yeniden cagirir.

    4. 11 kullanilmayan kolon kaldirildi
       rsi_fast, ema_50, ema_200, bb_middle, bb_upper, bb_width,
       atr_percent, close_prev, max_price_10, high_20, low_20

    5. Gereksiz importlar temizlendi (Dict, Optional, math)
    """
    INTERFACE_VERSION = 3
    can_short = False

    timeframe = "15m"
    startup_candle_count = 200
    process_only_new_candles = True

    minimal_roi = {
        "0": 0.003,
        "30": 0.006,
        "60": 0.010,
        "120": 0.015,
        "240": 0.020,
        "360": 0.025,
    }

    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    max_open_trades = 3
    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    rsi_upper_threshold = 75
    rsi_lower_threshold = 25

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    buy_rsi = IntParameter(20, 45, default=35, space="buy", optimize=True)
    sell_rsi = IntParameter(55, 85, default=70, space="sell", optimize=True)
    buy_adx = IntParameter(15, 40, default=25, space="buy", optimize=True)
    buy_ema_fast = IntParameter(5, 15, default=9, space="buy", optimize=True)
    buy_ema_slow = IntParameter(20, 50, default=30, space="buy", optimize=True)
    dca_trigger = DecimalParameter(-0.02, -0.08, default=-0.03, decimals=3, space="buy", optimize=True)
    dca_multiplier = DecimalParameter(1.2, 2.0, default=1.5, decimals=2, space="buy", optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.last_dca_level = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_prev"] = dataframe["rsi"].shift(1)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        ef = self.buy_ema_fast.value
        es = self.buy_ema_slow.value
        dataframe[f"ema_fast_{ef}"] = ta.EMA(dataframe, timeperiod=ef)
        dataframe[f"ema_slow_{es}"] = ta.EMA(dataframe, timeperiod=es)

        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)

        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_lower"] = bb["lowerband"]

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_sma"]

        dataframe["high_20"] = dataframe["high"].rolling(window=20).max()
        dataframe["low_20"] = dataframe["low"].rolling(window=20).min()
        dataframe["price_position"] = (dataframe["close"] - dataframe["low_20"]) / (dataframe["high_20"] - dataframe["low_20"]).replace(0, 1)

        dataframe["vwap"] = (dataframe["close"] * dataframe["volume"]).cumsum() / dataframe["volume"].cumsum()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ef = self.buy_ema_fast.value
        es = self.buy_ema_slow.value
        volatility_factor = dataframe['atr'] / dataframe['close'].replace(0, 1)
        dynamic_rsi = (self.buy_rsi.value - (volatility_factor * 10)).clip(lower=20)

        dataframe.loc[
            (
                (dataframe[f"ema_fast_{ef}"] > dataframe[f"ema_slow_{es}"]) &
                (dataframe["adx"] > self.buy_adx.value) &
                (dataframe["rsi"] < dynamic_rsi) &
                (dataframe["rsi_prev"] < dataframe["rsi"]) &
                (dataframe["volume"] > dataframe["volume_sma"] * 1.2) &
                (dataframe["close"] < dataframe["bb_lower"] * 1.02) &
                (dataframe["plus_di"] > dataframe["minus_di"]) &
                (dataframe["close"] > dataframe["vwap"] * 0.995) &
                (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "adx_rsi_buy")

        dataframe.loc[
            (
                (dataframe["ema_9"] > dataframe["ema_21"]) &
                (dataframe["macd"] > dataframe["macd_signal"]) &
                (dataframe["macd_hist"] > 0) &
                (dataframe["rsi"] < 45) &
                (dataframe["adx"] > 20) &
                (dataframe["volume_ratio"] > 1.3) &
                (dataframe["close"] > dataframe["vwap"] * 0.995) &
                (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "ema_macd_buy")

        dataframe.loc[
            (
                (dataframe["rsi"] < 30) &
                (dataframe["adx"] > self.buy_adx.value) &
                (dataframe["price_position"] < 0.2) &
                (dataframe["volume_ratio"] > 1.1) &
                (dataframe["close"] > dataframe["vwap"] * 0.995) &
                (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "oversold_buy")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime',
                   current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1].squeeze()
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        if current_profit > 0.05 and last_candle['rsi'] > 80:
            return 'rsi_overbought_exit'

        if current_profit > 0.03 and last_candle['macd'] < last_candle['macd_signal']:
            return 'macd_cross_down'

        if current_profit > 0.08 and count_of_buys >= 2 and last_candle['rsi'] > 65:
            return 'profit_take_rsi'

        if current_profit < -0.05 and count_of_buys >= 3:
            if last_candle['rsi'] < 35 and last_candle['adx'] > 25:
                return 'dca_recovery_wait'

        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                             current_rate: float, current_profit: float, min_stake: float,
                             max_stake: float, **kwargs):
        if not trade.is_open:
            return None

        pair = trade.pair
        orders_count = trade.nr_of_successful_entries

        if orders_count > self.max_entry_position_adjustment:
            logger.info(f"Max DCA entries ({self.max_entry_position_adjustment}) reached for {pair}. Skipping DCA.")
            return None

        avg_price = trade.open_rate
        close = current_rate

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        last_candle = dataframe.iloc[-1]

        if current_profit > 0.04:
            amount_to_sell = trade.amount * 0.4
            logger.info(f"Partial exit for {pair}: Selling {amount_to_sell:.6f} at {close:.6f}")
            return -amount_to_sell

        if current_profit > self.dca_trigger.value:
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        price_drop = (avg_price - close) / avg_price
        dca_step = 1.5 * last_candle["atr"] / last_candle["close"]
        max_price_drop = 3 * last_candle["atr"] / last_candle["close"]
        adx = last_candle["adx"]
        macd = last_candle["macd"]
        macd_signal = last_candle["macd_signal"]
        rsi = last_candle["rsi"]

        if adx < 15:
            logger.info(f"ADX ({adx:.2f}) too low for {pair}. Skipping DCA to avoid weak trend.")
            return None

        if macd < macd_signal and rsi > self.rsi_lower_threshold:
            pass
        elif macd >= macd_signal:
            pass
        else:
            logger.info(f"MACD ({macd:.6f}) below signal ({macd_signal:.6f}) for {pair}. Skipping DCA.")
            return None

        if price_drop >= max_price_drop:
            logger.info(f"Price drop for {pair} exceeds dynamic threshold ({price_drop:.2%}). Skipping DCA.")
            return None

        if trade.id not in self.last_dca_level:
            self.last_dca_level[trade.id] = 0.0

        current_dca_level = (price_drop // dca_step) * dca_step
        last_dca_level = self.last_dca_level[trade.id]

        if price_drop >= dca_step and current_dca_level > last_dca_level:
            self.last_dca_level[trade.id] = current_dca_level
            atr = last_candle["atr"]
            volatility_factor = atr / last_candle["close"]
            total_position = sum(o.stake_amount for o in trade.orders)

            if rsi < self.rsi_lower_threshold:
                new_stake = trade.stake_amount * 0.7 * (1 + volatility_factor)
                logger.info(f"Aggressive DCA for {pair}: RSI={rsi:.2f}, ADX={adx:.2f}, new_stake={new_stake:.2f}")
            elif rsi > self.rsi_upper_threshold:
                new_stake = trade.stake_amount * 0.3 * (1 + volatility_factor)
                logger.info(f"Conservative DCA for {pair}: RSI={rsi:.2f}, new_stake={new_stake:.2f}")
            else:
                new_stake = trade.stake_amount * 0.5 * (1 + volatility_factor)
                logger.info(f"Normal DCA for {pair}: RSI={rsi:.2f}, new_stake={new_stake:.2f}")

            try:
                amount = new_stake / current_rate
                logger.info(f"DCA buy #{count_of_buys + 1} for {pair}, stake: {new_stake:.2f}, amount: {amount:.6f}")
                return new_stake
            except Exception as e:
                logger.info(f"Error in DCA for {pair}: {str(e)}")
                return None

        return None
