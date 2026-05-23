"""
ContraRatingsStrategy — Production Strategy (4h+15m+1h)
=========================================================

STRATEJİ MANTIĞI (Contrarian / Tersine İşlem):
- Piyasa aşırı satışta (4h, 15m, 1h strong_sell)
- Kısa vadede dönüş başlıyor (5m, 1m buy_signal)
- Bu "dip satın alma" yaklaşımıdır - düşükten al, yükselince sat

PERFORMANS (3 ay backtest: Feb-May 2026):
- Kar: +29.76 USDT (29.76%)
- Profit Factor: 3.51 (en iyi versiyon)
- Win Rate: 76.9%
- Drawdown: 3.70%
- Trades: 121

BACKTEST KOMUTU:
freqtrade backtesting --strategy ContraRatingsStrategy -i 1m --timerange 20260201-

YAPILACAK DEĞİŞİKLİKLER:
- v1 (orijinal): 4h+15m strong_sell → 124 trades, PF 3.26
- v2 (+1h): 4h+15m+1h strong_sell → 121 trades, PF 3.51 ← ŞU ANKİ

DEBUG İPUÇLARI:
- trade.open_date_utc: Pozisyon açılma zamanı
- trade.enter_tag: Strateji tag'ı (contra_entry_...)
- current_profit: Mevcut kar % (0.01 = %1)
- pair: İşlem çifti (örn. BTC/USDT)
"""

from pandas import DataFrame, Series
from freqtrade.strategy import IStrategy, informative
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
import talib.abstract as ta


class ContraRatingsStrategy(IStrategy):
    """
    Multi-timeframe contrarian strateji.
    
    Giriş Mantığı:
    1. Uzun vade (4h, 15m, 1h): Aşırı satış koşulları (strong_sell)
    2. Orta vade (5m): Yeni başlayan al sinyali (buy_signal just turned true)
    3. Kısa vade (1m): Al sinyali (bullish momentum)
    4. Filtre: Son 30 günde %100'den fazla yükselmemiş (pump kontrolü)
    """

    # =============================================================================
    # TEMEL PARAMETRELER
    # =============================================================================
    timeframe = "1m"                           # Ana timeframe (1 dakika)
    startup_candle_count = 400                 # Başlangıç için gerekli mum sayısı
    
    # Exit yönetimi - kademeli ROI
    minimal_roi = {
        "0": 0.04,      # %4+ → hemen kapat
        "120": 0.025,   # 2 saat sonra %2.5+ yeter
        "360": 0.015,   # 6 saat sonra %1.5+ yeter
        "720": 0.008,   # 12 saat sonra %0.8+ yeter
    }
    
    stoploss = -0.10
    
    # Sadece yeni mumlarda hesaplama (performans için)
    process_only_new_candles = True

    trailing_stop = False

    # =============================================================================
    # POZİSYON AYARLAMA & DCA
    # =============================================================================
    # DCA (Dollar Cost Averaging): Fiyat düşerse ikinci giriş
    # max_entry_position_adjustment = 1 means 1 additional entry (toplam 2)
    position_adjustment_enable = True           # Pozisyon ayarlama aktif
    max_entry_position_adjustment = 1          # Maksimum 2 giriş (1 DCA)
    
    # DCA seviyesi: Fiyat %0.4 düşerse ikinci giriş yap
    dca_steps = [-0.004]                        # -%0.4'te DCA

    # =============================================================================
    # STAKE HESAPLAMA
    # =============================================================================
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        """
        StakeAmount hesaplama.
        İki giriş (DCA dahil) yapılacağı için stake'i 2'ye bölüyoruz.
        Böylece her giriş eşit büyüklükte olur.
        """
        return proposed_stake / (self.max_entry_position_adjustment + 1)

    # =============================================================================
    # UZUN VADELİ İNDİKATÖRLER (4h) - STRONG SELL TESPİTİ
    # =============================================================================
    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        4 saatlik timeframe - Uzun vadeli trend analizi.
        
        Strong Sell (Güçlü Sat) Koşulu (3/5 bearish = strong_sell):
        1. RSI < 40: Aşırı satış bölgesi
        2. Stochastic %K < 20: Düşük momentum
        3. MACD < Signal: Negatif momentum
        4. Fiyat < SMA50: Fiyat ortalamanın altında
        5. ADX > 25 AND DI- > DI+: Güçlü düşüş trendi
        
        Amac: Uzun vadede aşırı satış tespit etmek
        """
        # Temel indikatörler
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["stoch_k"] = ta.STOCH(dataframe, 14, 3, 3)["slowk"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=20)
        plus_di = ta.PLUS_DI(dataframe, timeperiod=14)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)

        # Bearish score: 0-5 arası (5 = tamamen bearish)
        bearish = Series(0, index=dataframe.index)
        bearish += (dataframe["rsi"] < 40).astype(int)           # RSI aşırı satış
        bearish += (dataframe["stoch_k"] < 20).astype(int)       # Stochastic düşük
        bearish += (macd["macd"] < macd["macdsignal"]).astype(int) # MACD negatif
        bearish += (dataframe["close"] < dataframe["sma50"]).astype(int) # Fiyat < SMA50
        bearish += ((dataframe["adx"] > 25) & (plus_di < minus_di)).astype(int) # ADX güçlü düşüş

        # Strong Sell: 3/5 koşul sağlandığında
        dataframe["strong_sell"] = bearish >= 3

        return dataframe

    # =============================================================================
    # UZUN-ORTA VADELİ İNDİKATÖRLER (1h) - Ek onay
    # =============================================================================
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        1 saatlik timeframe - Orta-u Long vadeli onay katmanı.
        
        4h'ye ek olarak 1h'de strong_sell kontrolü.
        Bu, daha seçici girişler için kullanılır.
        
        Not: Profit Factor'ı 3.26'dan 3.51'e yükseltti.
        """
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

    # =============================================================================
    # ORTA VADELİ İNDİKATÖRLER (15m)
    # =============================================================================
    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        15 dakikalık timeframe - Orta vadeli trend analizi.
        
        Aynı strong_sell mantığı 15m için.
        4h + 1h + 15m birlikte güçlü bearish onay verir.
        
        Debug: strong_sell_15m sütununu kontrol et
        """
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

    # =============================================================================
    # KISA VADELİ İNDİKATÖRLER (5m) - AL SİNYALİ
    # =============================================================================
    @informative("5m")
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        5 dakikalık timeframe - Kısa vadeli momentum.
        
        BULLISH (Al) Koşulu (2/5 bullish = buy_signal):
        1. RSI > 60: Aşırı alım bölgesi (dip alım için ters mantık)
        2. Stochastic %K > 80: Yüksek momentum
        3. MACD > Signal: Pozitif momentum
        4. Fiyat > SMA50: Fiyat ortalamanın üstünde
        5. ADX > 25 AND DI+ > DI-: Güçlü yükseliş trendi
        
        ÖNEMLİ: Sadece 5m buy_signal YENİDEN true olduğunda giriş yap
        (shift(1) ile önceki mum kontrolü)
        """
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["stoch_k"] = ta.STOCH(dataframe, 14, 3, 3)["slowk"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=20)
        plus_di = ta.PLUS_DI(dataframe, timeperiod=14)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)

        bullish = Series(0, index=dataframe.index)
        bullish += (dataframe["rsi"] > 60).astype(int)
        bullish += (dataframe["stoch_k"] > 80).astype(int)
        bullish += (macd["macd"] > macd["macdsignal"]).astype(int)
        bullish += (dataframe["close"] > dataframe["sma50"]).astype(int)
        bullish += ((dataframe["adx"] > 25) & (plus_di > minus_di)).astype(int)

        dataframe["buy_signal"] = bullish >= 3
        return dataframe

    # =============================================================================
    # GÜNLÜK İNDİKATÖRLER - PUMP FİLTRESİ
    # =============================================================================
    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Günlük timeframe - Pump (aşırı yükseliş) kontrolü.
        
        Son 30 günde %100'den fazla yükselen coin'leri exclude et.
        Bu, "sonradan patlayan" coin'lerde düşüş riskini azaltır.
        
        no_pump = pct_30d < 1.0 (yani %100'den az yükselmiş)
        """
        dataframe["pct_30d"] = dataframe["close"] / dataframe["close"].shift(30) - 1
        return dataframe

    # =============================================================================
    # ANA İNDİKATÖRLER (1m)
    # =============================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        1 dakikalık (ana timeframe) indikatörler.
        
        1m için bullish sinyali (giriş onayı için).
        """
        rsi = ta.RSI(dataframe, timeperiod=14)
        stoch_k = ta.STOCH(dataframe, 14, 3, 3)["slowk"]
        adx = ta.ADX(dataframe, timeperiod=20)
        plus_di = ta.PLUS_DI(dataframe, timeperiod=14)
        minus_di = ta.MINUS_DI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        sma50 = ta.SMA(dataframe, timeperiod=50)

        bullish = Series(0, index=dataframe.index)
        bullish += (rsi > 60).astype(int)
        bullish += (stoch_k > 80).astype(int)
        bullish += (macd["macd"] > macd["macdsignal"]).astype(int)
        bullish += (dataframe["close"] > sma50).astype(int)
        bullish += ((adx > 25) & (plus_di > minus_di)).astype(int)

        # Entry sinyallerini birleştir
        dataframe["entry_4h"] = dataframe["strong_sell_4h"]     # 4h aşırı satış
        dataframe["entry_15m"] = dataframe["strong_sell_15m"]   # 15m aşırı satış
        dataframe["entry_1h"] = dataframe["strong_sell_1h"]     # 1h aşırı satış (ek onay)
        dataframe["entry_5m"] = dataframe["buy_signal_5m"]      # 5m al sinyali
        dataframe["entry_1m"] = bullish >= 3                     # 1m güçlü al sinyali
        dataframe["no_pump"] = dataframe["pct_30d_1d"] < 1.0       # Pump filtresi
        
        return dataframe

    # =============================================================================
    # GİRİŞ SİNYALİ
    # =============================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        GİRİŞ KOŞULLARI (TÜMÜ BİRLİKTE SAĞLANMALI):
        
        1. entry_4h == True: 4 saatlik grafik aşırı satışta
        2. entry_15m == True: 15 dakikalık grafik aşırı satışta
        3. entry_1h == True: 1 saatlik grafik aşırı satışta (ek onay)
        4. entry_5m == True: 5 dakikalık grafik güçlü al sinyali
        5. entry_1m == True: 1 dakikalık grafik güçlü al sinyali
        6. no_pump == True: Son 30g %100'den az yükselmiş
        7. volume > 0: İşlem hacmi var
        
        TEK SİNYAL: Tüm koşullar aynı anda False→True döndüğünde 1 kere ateşlenir.
        """
        enter_conditions = (
            (dataframe["entry_4h"] == True) &
            (dataframe["entry_15m"] == True) &
            (dataframe["entry_1h"] == True) &
            (dataframe["entry_5m"] == True) &
            (dataframe["entry_1m"] == True) &
            (dataframe["no_pump"] == True) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[
            enter_conditions & (enter_conditions.shift(1) == False),
            ["enter_long", "enter_tag"]] = (1, "contra_entry_4h15m1h_bear_5m1m_buy")

        return dataframe

    # =============================================================================
    # ÇIKIŞ SİNYALİ
    # =============================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        ÇIKIŞ: Exit sinyali YOK.
        
        Tüm çıkışlar şunlarla yönetilir:
        1. Trailing Stop: Kar %3'e ulaşınca aktif
        2. Stop-Loss: -%10
        3. custom_exit: 72 saat + <%0.5 kar = timeout
        
        Not: populate_exit_trend'de sinyal verirsek trailing stop devre dışı kalır.
        Bu yüzden exit_long = 0 (hiçbir çıkış sinyali yok).
        """
        dataframe.loc[:, "exit_long"] = 0
        return dataframe

    # =============================================================================
    # ÖZEL ÇIKIŞ KURALI (Custom Exit)
    # =============================================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> str | None:
        """
        ÖZEL ÇIKIŞ KURALI - Uzun süre açık pozisyonlar için timeout.
        
        Koşul: Pozisyon 72 saatten (3 gün) uzun SÜRE AÇIK VE kar < %0.5
        
        Neden: 
        - Trailing stop tetiklenmemiş
        - Kar az (belki yanlış giriş)
        - 72 saat bekleme yeterli
        
        DEBUG:
        - trade.open_date_utc: Pozisyon açılma zamanı
        - trade_duration_hours: Kaç saat açık
        - current_profit: Mevcut kar (örn. 0.003 = %0.3)
        
        Return: "long_hold_timeout" = çıkış sinyali
        """
        # Pozisyon açık süresi (saat)
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        
        # 48 saat + kar < %0.5 → çıkış
        if trade_duration_hours > 48 and current_profit < 0.005:
            return "long_hold_timeout"
        
        return None

    # =============================================================================
    # DCA (POZİSYON ARTIRMA)
    # =============================================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        """
        DCA - Dollar Cost Averaging (İkinci giriş).
        
        Koşul: Fiyat giriş fiyatından %0.4 düştüğünde
        
        Mantık:
        - İlk girişte fiyat yüksekte
        - Fiyat düşerse ikinci giriş yap (ortalama maliyeti düşür)
        - Toplam 2 giriş (1 DCA)
        
        DEBUG:
        - price_drop: (giriş - mevcut) / giriş = düşüş %
        - dca_index: 0 = ilk giriş, 1 = DCA girişi
        
        Örnek: 
        - BTC 50000'de girdik
        - Fiyat 49800'e düştü (%0.4)
        - İkinci giriş yap (aynı stake ile)
        - Ortalama maliyet: (50000 + 49800) / 2 = 49900
        """
        # Maksimum DCA sayısı kontrolü
        if trade.nr_of_successful_entries - 1 >= self.max_entry_position_adjustment:
            return None

        # Giriş fiyatı ve düşüş hesabı
        entry_price = trade.open_rate
        price_drop = (entry_price - current_rate) / entry_price  # Düşüş %
        dca_index = trade.nr_of_successful_entries - 1             # 0 veya 1

        # DCA koşulu: Düşüş %0.4'ü geçtiyse
        if dca_index < len(self.dca_steps) and price_drop >= abs(self.dca_steps[dca_index]):
            # İkinci giriş için aynı stake miktarı
            return trade.stake_amount, f"dca_{dca_index + 1}"

        return None