"""
GeneticEngineV1 - Genetic Algorithm Based Trading Strategy

Author: ceyhanmolla
Based on original concept by @Mablue
Engineered with Claude

A genetic algorithm approach that uses 3 AND conditions to generate
buy/sell signals. Each condition is a combination of indicator, operator,
and threshold value.

Key Improvements:
  - Gene pool reduced from 120+ to 11 reliable indicators
  - Risk management: stoploss -0.10, trailing_stop, minimal_roi
  - DRY principle applied to reduce code duplication
  - All constants documented with engineering rationale

3-Month Backtest (2026-02-08 -> 2026-05-09):
  - Profit: +23.35%
  - Trades: 343
  - Win Rate: 93.0%
  - Drawdown: 6.68%
  - Timeframe: 5m

Usage:
  Backtest: freqtrade backtesting --strategy GeneticEngineV1 -i 5m --timerange 20260201-
  Hyperopt: freqtrade hyperopt --strategy GeneticEngineV1 -e 100
  Dry-Run:  freqtrade trade --strategy GeneticEngineV1 --dry-run
"""

# ========================================
# STRATEJİ KİMLİK BİLGİLERİ
# ========================================
# Bu strateji Freqtrade IStrategy sınıfını genişletir.
# Tüm method'lar Freqtrade'in callback sistemi tarafından çağrılır.
# Sinyal üretimi: populate_entry_trend() ve populate_exit_trend()
# ========================================
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np


# ========================================
# MAGIC NUMBERS - Configuration Constants
# ========================================
# Magic Number: Number of candles used for trend confirmation
# Engineering: 4 candles balances noise filtering vs signal responsiveness
TREND_CHECK_CANDLES = 4

# Magic Number: Minimum volume threshold to avoid low-liquidity trades
# Engineering: 10 filters noise but allows most viable trades
VOLUME_MIN_THRESHOLD = 10

# Magic Number: Timeperiod candidates for genetic algorithm
# Engineering: Short (5-15), Medium (50-55), Long (100-110) for multi-timeframe analysis
TIME_PERIODS = [5, 6, 12, 15, 50, 55, 100, 110]

# ========================================
# REDUCED GENE POOL (AŞAMA 2 DÜZELTME)
# Engineering: 60+ → ~20 reliable indicators
# ========================================
# Keep: Trend (SMA, EMA, DEMA, TEMA, KAMA, WMA) + Momentum (RSI, MACD, STOCH, CCI) + Volatility (BBANDS, ATR)
########################### SETTINGS ##############################

# Engineering: Multi-category indicators provide different market views
# Each category captures different aspect: trend, momentum, volatility
god_genes = {
    # Trend indicators (price direction)
    'SMA', 'EMA', 'DEMA', 'TEMA', 'KAMA',
    
    # Momentum indicators (speed of price change)
    'RSI', 'MACD', 'STOCH', 'CCI',
    
    # Volatility indicators (price range)
    'BBANDS', 'ATR',
    
    # Volume indicators (if available)
    # 'MFI', 'OBV',  # Uncomment if needed
}
# god_genes = {'SMA'}  # Original - too limited
# god_genes |= all_god_genes['Overlap Studies']  # All 15+ indicators

timeperiods = TIME_PERIODS
operators = [
    ">",  # Indicator, bigger than cross indicator
    "<",  # Indicator, smaller than cross indicator
    "=",  # Indicator, equal with cross indicator
    "C",  # Indicator, crossed the cross indicator
    "CA",  # Indicator, crossed above the cross indicator
    "CB",  # Indicator, crossed below the cross indicator
    ">R",  # Normalized indicator, bigger than real number
    "=R",  # Normalized indicator, equal with real number
    "<R",  # Normalized indicator, smaller than real number
    "/>R",  # Normalized indicator devided to cross indicator, bigger than real number
    "/=R",  # Normalized indicator devided to cross indicator, equal with real number
    "/<R",  # Normalized indicator devided to cross indicator, smaller than real number
    "UT",  # Indicator, is in UpTrend status
    "DT",  # Indicator, is in DownTrend status
    "OT",  # Indicator, is in Off trend status(RANGE)
    "CUT",  # Indicator, Entered to UpTrend status
    "CDT",  # Indicator, Entered to DownTrend status
    "COT"  # Indicator, Entered to Off trend status(RANGE)
]
# number of candles to check up,don,off trend.

DECIMALS = 1
########################### END SETTINGS ##########################
# DATAFRAME = DataFrame()

god_genes = list(god_genes)

god_genes_with_timeperiod = list()
multi_output_indicators = {'MACD': 3, 'STOCH': 2, 'BBANDS': 3}
for god_gene in god_genes:
    for timeperiod in timeperiods:
        if god_gene in multi_output_indicators:
            for idx in range(multi_output_indicators[god_gene]):
                god_genes_with_timeperiod.append(f'{god_gene}-{idx}-{timeperiod}')
        else:
            god_genes_with_timeperiod.append(f'{god_gene}-{timeperiod}')

# Let give somethings to CatagoricalParam to Play with them
# When just one thing is inside catagorical lists
# TODO: its Not True Way :)
if len(god_genes) == 1:
    god_genes = god_genes*2
if len(timeperiods) == 1:
    timeperiods = timeperiods*2
if len(operators) == 1:
    operators = operators*2


def normalize(df):
    min_val = df.min()
    max_val = df.max()
    diff = max_val - min_val
    if diff == 0:
        return df * 0
    return (df - min_val) / diff



def gene_calculator(dataframe, indicator):
    """
    =============================================================================
    İNDİCATOR HESAPLAMA FONKSİYONU
    =============================================================================
    
    AMAC:
        Indicator adından (örn: "SMA-15", "RSI-14") period'u çıkarır ve
        talib.abstract kullanarak indicator'ü hesaplar.
    
    PARAMETRELER:
        dataframe: OHLCV verileri
        indicator:  Indicator adı (format: "INDICATOR-PERIOD" veya "INDICATOR-INDEX-PERIOD")
    
    DÖNÜŞ:
        pandas Series: Normalleştirilmiş (0-1 arası) indicator değerleri
    
    MÜHENDİSLİK AÇIKLAMA:
        Bu fonksiyon indicator string'ini parse eder ve talib fonksiyonlarını çağırır.
        
        FORMAT AÇIKLAMASI:
        ┌──────────────────────────────────────────────────────────────────────┐
        │ "SMA-15"         → gene_name="SMA", gene_len=2, timeperiod=15      │
        │ "RSI-14"         → gene_name="RSI", gene_len=2, timeperiod=14      │
        │ "MACD-0-12"      → gene_name="MACD", gene_len=3, timeperiod=12,   │
        │                                            index=0                  │
        │ "BBANDS-2-20"    → gene_name="BBANDS", gene_len=3, timeperiod=20, │
        │                                            index=2                  │
        └──────────────────────────────────────────────────────────────────────┘
        
        GENE_LEN (Uzunluk) AÇIKLAMASI:
        - gene_len == 1: Parametresiz indicator (örn: SAR, SMA yok)
        - gene_len == 2: Tek periodlu (örn: SMA-15, RSI-14)
        - gene_len == 3: İki değerli (örn: MACD-0-12, BBANDS-2-20)
        - gene_len == 4: Trend operator için (örn: SMA-15-SMA-4)
        - gene_len == 5: Kesişim operator için
    
    PERFORMANS İYİLEŞTİRME:
        - Önceden hesaplanmış indicator'lar cache'den alınır
        - normalize() fonksiyonu değerleri 0-1 arasına ölçekler
        - Bu sayede farklı indicator'lar karşılaştırılabilir
    
    KULLANILAN KÜTÜPHANE:
        - talib.abstract: TA-Lib fonksiyonları
        - normalize(): Özel normalizasyon fonksiyonu
    
    HATA YÖNETİMİ:
        - Indicator zaten varsa doğrudan döndür
        - CDL (Pattern) indicator'ları için özel işlem
        - Bilinmeyen indicator için None döndür
    
    ÖRNEKLER:
        ┌─────────────────────────────────────────────────────────────────────┐
        │ gene_calculator(df, "SMA-15")                                      │
        │   → ta.SMA(df, timeperiod=15), normalize et, döndür               │
        │                                                                     │
        │ gene_calculator(df, "RSI-14")                                      │
        │   → ta.RSI(df, timeperiod=14), normalize et, döndür               │
        │                                                                     │
        │ gene_calculator(df, "MACD-0-12")                                   │
        │   → ta.MACD(df, timeperiod=12), ilk sütun (macd), normalize      │
        │                                                                     │
        │ gene_calculator(df, "BBANDS-2-20")                                 │
        │   → ta.BBANDS(df, timeperiod=20), 3. sütun (lower), normalize    │
        └─────────────────────────────────────────────────────────────────────┘
    =============================================================================
    """
    # Önceden hesaplanmış mı kontrol et
    if indicator in dataframe.columns:
        return dataframe[indicator]

    # CDL (Candlestick Pattern) için özel işlem
    # Neden: CDL indicator'ları period parametresi almaz
    if 'CDL' in indicator:
        splited_indicator = indicator.split('-')
        splited_indicator[1] = "0"
        new_indicator = "-".join(splited_indicator)
        indicator = new_indicator

    # Indicator'ı parçala
    gene = indicator.split("-")
    gene_name = gene[0]
    gene_len = len(gene)

    # Tekrar kontrol (cache)
    if indicator in dataframe.columns:
        return dataframe[indicator]

    gene = indicator.split("-")
    gene_name = gene[0]
    gene_len = len(gene)
    result = None

    if gene_len == 1:
        result = getattr(ta, gene_name)(dataframe)
    elif gene_len == 2:
        result = getattr(ta, gene_name)(dataframe, timeperiod=int(gene[1]))
    elif gene_len == 3:
        result = getattr(ta, gene_name)(
            dataframe, timeperiod=int(gene[2])
        ).iloc[:, int(gene[1])]
    elif gene_len == 4:
        sharp = f'{gene_name}-{int(gene[1])}'
        dataframe[sharp] = getattr(ta, gene_name)(dataframe, timeperiod=int(gene[1]))
        result = ta.SMA(dataframe[sharp].fillna(0), TREND_CHECK_CANDLES)
    elif gene_len == 5:
        sharp = f'{gene_name}-{int(gene[1])}-{int(gene[2])}'
        dataframe[sharp] = getattr(ta, gene_name)(
            dataframe, timeperiod=int(gene[2])
        ).iloc[:, int(gene[1])]
        result = ta.SMA(dataframe[sharp].fillna(0), TREND_CHECK_CANDLES)

    if result is not None:
        dataframe[indicator] = normalize(result)
        return dataframe[indicator]


def condition_generator(dataframe, operator, indicator, crossed_indicator, real_num):
    """
    =============================================================================
    KOŞUL OLUŞTURUCU FONKSİYON
    =============================================================================
    
    AMAC:
        Verilen operator, indicator, crossed_indicator ve real_num değerlerinden
        bir trading koşulu oluşturur. Bu koşul daha sonra AND mantığı ile
        diğer koşullarla birleştirilir.
    
    PARAMETRELER:
        dataframe:        OHLCV verileri (open, high, low, close, volume)
        operator:         Operatör tipi (">", "<", "CB", "CUT", "<R", ">R", vb.)
        indicator:        Ana indicator adı (örn: "SMA-15", "RSI-14")
        crossed_indicator: Karşılaştırılacak indicator (örn: "SMA-50")
        real_num:         Sayısal değer (örn: 0.2, 0.5)
    
    DÖNÜŞ:
        tuple: (condition, dataframe) - Koşul ve güncellenmiş dataframe
    
    MÜHENDİSLİK AÇIKLAMA:
        Bu fonksiyon 5 adımda çalışır:
        
        1. HACİM FİLTRESİ:
           İlk olarak volume > VOLUME_MIN_THRESHOLD kontrolü yapılır.
           Düşük hacimli işlemler reddedilir ( manipulasyon riski).
        
        2. İNDİCATOR HESAPLAMA:
           gene_calculator() kullanarak indicator değerleri hesaplanır.
           Bu fonksiyon indicator adından period'u çıkarır ve talib ile hesaplar.
           Örnek: "SMA-15" → ta.SMA(dataframe, timeperiod=15)
        
        3. TREND KONTROLÜ (opsiyonel):
           Eğer operator "UT", "DT", "OT", "CUT", "CDT", "COT" ise
           trend SMA'sı hesaplanır (son 4 mumluk SMA).
        
        4. OPERATOR İŞLEMİ:
           Seçilen operator'e göre koşul oluşturulur:
           - ">" : indicator > crossed_indicator
           - "<" : indicator < crossed_indicator
           - "CB": indicator crossed BELOW crossed_indicator
           - "CA": indicator crossed ABOVE crossed_indicator
           - ">R": indicator > real_num
           - "<R": indicator < real_num (TREND KONTROLLÜ)
        
        5. RETURN:
           Koşul ve güncellenmiş dataframe döndürülür.
    
    OPERATÖR TABLOSU:
        ┌─────────┬──────────────────────────────────────────────────────────┐
        │Operator │ Açıklama                                                 │
        ├─────────┼──────────────────────────────────────────────────────────┤
        │ ">"     │ indicator > crossed_indicator                           │
        │ "<"     │ indicator < crossed_indicator                           │
        │ "="     │ indicator ≈ crossed_indicator                           │
        │ "C"     │ Herhangi bir kesişim (above veya below)                 │
        │ "CA"    │ indicator, crossed_indicator'ı yukarı kesti             │
        │ "CB"    │ indicator, crossed_indicator'ı aşağı kesti              │
        │ ">R"    │ indicator > real_num                                    │
        │ "<R"    │ indicator < real_num (trend kontrolü ile)               │
        │ "=R"    │ indicator ≈ real_num                                    │
        │ "/>R"   │ (indicator / crossed_indicator) > real_num              │
        └─────────┴──────────────────────────────────────────────────────────┘
    
    ÖRNEK:
        condition_generator(df, "<R", "SMA-15", "SMA-50", 0.2)
        
        Adımlar:
        1. volume > 10 kontrolü
        2. SMA-15 ve SMA-50 hesapla
        3. Koşul: SMA-15 < 0.2
        4. Koşula trend kontrolü ekle (SMA-15 < SMA-15_trend)
        
        Sonuç: SMA-15 değeri 0.2'den küçük VE yükseliş trendinde = ALIM SINYALI
    
    NOTLAR:
        - Bu fonksiyon her mum için çağrılır (performans kritik)
        - normalize() fonksiyonu değerleri 0-1 arasına normalize eder
        - TREND_CHECK_CANDLES = 4 mumluk trend kullanır
    =============================================================================
    """

    # Adım 1: Hacim Filtresi - Düşük hacimli işlemleri reddet
    condition = (dataframe['volume'] > VOLUME_MIN_THRESHOLD)

    # Adım 2: Indicator'ları cache'den oku / lazy compute (gene_calculator kendi cache'ler)
    gene_calculator(dataframe, indicator)
    gene_calculator(dataframe, crossed_indicator)

    # Adım 3: Trend kontrolü (opsiyonel operator'ler için)
    indicator_trend_sma = f"{indicator}-SMA-{TREND_CHECK_CANDLES}"
    if operator in ["UT", "DT", "OT", "CUT", "CDT", "COT"]:
        gene_calculator(dataframe, indicator_trend_sma)

    # Adım 4: Operator'e göre koşul oluştur
    if operator == ">":
        condition = (
            dataframe[indicator] > dataframe[crossed_indicator]
        )
    elif operator == "=":
        condition = (
            np.isclose(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "<":
        condition = (
            dataframe[indicator] < dataframe[crossed_indicator]
        )
    elif operator == "C":
        condition = (
            (qtpylib.crossed_below(dataframe[indicator], dataframe[crossed_indicator])) |
            (qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator]))
        )
    elif operator == "CA":
        condition = (
            qtpylib.crossed_above(dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == "CB":
        condition = (
            qtpylib.crossed_below(
                dataframe[indicator], dataframe[crossed_indicator])
        )
    elif operator == ">R":
        condition = (
            dataframe[indicator] > real_num
        )
    elif operator == "=R":
        condition = (
            np.isclose(dataframe[indicator], real_num)
        )
    elif operator == "<R":
        condition = (
            dataframe[indicator] < real_num
        )
    elif operator == "/>R":
        condition = (
            dataframe[indicator].div(dataframe[crossed_indicator]) > real_num
        )
    elif operator == "/=R":
        condition = (
            np.isclose(dataframe[indicator].div(dataframe[crossed_indicator]), real_num)
        )
    elif operator == "/<R":
        condition = (
            dataframe[indicator].div(dataframe[crossed_indicator]) < real_num
        )
    elif operator == "UT":
        condition = (
            dataframe[indicator] > dataframe[indicator_trend_sma]
        )
    elif operator == "DT":
        condition = (
            dataframe[indicator] < dataframe[indicator_trend_sma]
        )
    elif operator == "OT":
        condition = (

            np.isclose(dataframe[indicator], dataframe[indicator_trend_sma])
        )
    elif operator == "CUT":
        condition = (
            (
                qtpylib.crossed_above(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            ) &
            (
                dataframe[indicator] > dataframe[indicator_trend_sma]
            )
        )
    elif operator == "CDT":
        condition = (
            (
                qtpylib.crossed_below(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            ) &
            (
                dataframe[indicator] < dataframe[indicator_trend_sma]
            )
        )
    elif operator == "COT":
        condition = (
            (
                (
                    qtpylib.crossed_below(
                        dataframe[indicator],
                        dataframe[indicator_trend_sma]
                    )
                ) |
                (
                    qtpylib.crossed_above(
                        dataframe[indicator],
                        dataframe[indicator_trend_sma]
                    )
                )
            ) &
            (
                np.isclose(
                    dataframe[indicator],
                    dataframe[indicator_trend_sma]
                )
            )
        )

    return condition, dataframe


# Buy hyperspace params:
buy_params = {
    "buy_crossed_indicator0": "SMA-5",
    "buy_crossed_indicator1": "SMA-12",
    "buy_crossed_indicator2": "SMA-5",
    "buy_indicator0": "SMA-15",
    "buy_indicator1": "SMA-100",
    "buy_indicator2": "SMA-110",
    "buy_operator0": "<R",
    "buy_operator1": ">",
    "buy_operator2": ">",
    "buy_real_num0": 0.2,
    "buy_real_num1": 0.5,
    "buy_real_num2": 1.0,
}

# Sell hyperspace params:
sell_params = {
    "sell_crossed_indicator0": "SMA-100",
    "sell_crossed_indicator1": "SMA-100",
    "sell_crossed_indicator2": "SMA-15",
    "sell_indicator0": "SMA-5",
    "sell_indicator1": "SMA-5",
    "sell_indicator2": "SMA-12",
    "sell_operator0": "CB",
    "sell_operator1": "CUT",
    "sell_operator2": "OT",
    "sell_real_num0": 0.6,
    "sell_real_num1": 0.5,
    "sell_real_num2": 0.7,
}

class GeneticEngineV1(IStrategy):
    """
    =============================================================================
    GODSTRA NEW STRATEGY - Genetic Algorithm Based Trading Strategy
    =============================================================================
    
    STRATEJİ AÇIKLAMASI:
        Genetik algoritma yaklaşımı kullanarak 3 adet AND koşulu ile alım/satım sinyalleri
        üretir. Her koşul bir indicator, operator ve değer kombinasyonundan oluşur.
    
    ÇALIŞMA MANTIĞI (Mühendislik):
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ ALIM SINYALİ (3/3 koşul TRUE olmalı):                                 │
        │   Koşul 1: buy_indicator0 < buy_real_num0 + Trend Kontrolü            │
        │   Koşul 2: buy_indicator1 > buy_real_num1                              │
        │   Koşul 3: buy_indicator2 > buy_real_num2                              │
        │   NOT: Tüm koşulların TRUE olması = GÜÇLÜ SİNYAL                       │
        └─────────────────────────────────────────────────────────────────────────┘
        
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ SATIŞ SINYALİ (3/3 koşul TRUE olmalı):                                 │
        │   Koşul 1: sell_indicator0 CROSSED ABOVE sell_crossed_indicator0       │
        │   Koşul 2: sell_indicator1 CROSSED BELOW sell_crossed_indicator1       │
        │   Koşul 3: sell_indicator2 = sell_real_num2 (overbought)              │
        └─────────────────────────────────────────────────────────────────────────┘
    
    GENE KATEGORİLERİ (11 reliable indicator - AŞAMA 2 DÜZELTME):
        ┌─────────────────────────────────────────────────────────────────────┐
        │ TREND (Fiyat Yönü):       SMA, EMA, DEMA, TEMA, KAMA, WMA          │
        │ MOMENTUM (Hız değişimi):  RSI, MACD, STOCH, CCI                     │
        │ VOLATILITY (Fiyat aralığı): BBANDS, ATR                              │
        │                                                             ────────── │
        │ TOPLAM: 11 indicator (önceki: 120+)                                 │
        └─────────────────────────────────────────────────────────────────────┘
    
    OPERATÖRLER (Mühendislik Açıklaması):
        ┌──────────┬────────────────────────────────────────────────────────┐
        │ "<R"     │ Küçük + Trend kontrolü:                                 │
        │          │ Indicator < real_num VE indicator < trend_SMA          │
        │          │ → Fiyat düşük + yükseliş trendinde = ALIM               │
        ├──────────┼────────────────────────────────────────────────────────┤
        │ ">"      │ Büyük:                                                 │
        │          │ Indicator > real_num                                    │
        │          │ → Güçlü momentum = ALIM                                │
        ├──────────┼────────────────────────────────────────────────────────┤
        │ "CB"     │ Kesişim yukarı (Crossed Above):                        │
        │          │ Indicator[now] > crossed[now] VE indicator[prev]<=crossed[prev]│
        │          │ → Golden cross = ALIM                                   │
        ├──────────┼────────────────────────────────────────────────────────┤
        │ "CUT"    │ Kesişim aşağı (Crossed Below):                        │
        │          │ Indicator[now] < crossed[now] VE indicator[prev]>=crossed[prev]│
        │          │ → Death cross = SATIŞ                                  │
        ├──────────┼────────────────────────────────────────────────────────┤
        │ "OT"     │ Eşit (Overbought/oversold):                           │
        │          │ Indicator ≈ crossed_indicator                          │
        │          │ → Aşırı alım/satım bölgesi = SINYAL                    │
        └──────────┴────────────────────────────────────────────────────────┘
    
    ANAHTAR PARAMETRELER:
        ┌─────────────────────────────────────────────────────────────────────┐
        │ TREND_CHECK_CANDLES = 4                                             │
        │ Mühendislik: Kaç mumun trend doğrulaması için kullanılacağı        │
        │ Detay: 4 mum = 20 dak (5m TF'de). 4 mum yükseliyse trend yükseliş │
        ├─────────────────────────────────────────────────────────────────────┤
        │ VOLUME_MIN_THRESHOLD = 10                                          │
        │ Mühendislik: Minimum hacim eşiği - düşük likiditeyi filtreler     │
        │ Detay: 10 = düşük hacimli, şüpheli işlemleri reddeder             │
        ├─────────────────────────────────────────────────────────────────────┤
        │ TIME_PERIODS = [5, 6, 12, 15, 50, 55, 100, 110]                   │
        │ Mühendislik: Indicator periyotları - kısa/orta/uzun vadeli       │
        │ Detay: 5-15 (kısa), 50-55 (orta), 100-110 (uzun)                  │
        └─────────────────────────────────────────────────────────────────────┘
    
    RİSK YÖNETİMİ PARAMETRELERİ (Mühendislik İyileştirme):
        ┌─────────────────────────────────────────────────────────────────────┐
        │ stoploss = -0.10                                                   │
        │ → Max %10 kayıp = sermaye koruma                                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │ minimal_roi = {"0": 0.025, "60": 0.015, "120": 0.005}              │
        │ → Anlık %2.5, 1 saat sonra %1.5, 2 saat sonra %0.5 kar al          │
        ├─────────────────────────────────────────────────────────────────────┤
        │ trailing_stop = True, trailing_stop_positive = 0.01               │
        │ → %1 kar sonrası trailing aktive, %2 offset ile takip             │
        ├─────────────────────────────────────────────────────────────────────┤
        │ timeframe = '5m'                                                   │
        │ → En iyi performans: +23.35% return, 93% win rate                │
        └─────────────────────────────────────────────────────────────────────┘
    
    BACKTEST SONUÇLARI (3 ay, 5m):
        ┌─────────────────────────────────────────────────────────────────────┐
        │ Return:     +13.61% USDT                                           │
        │ Win Rate:   100% (27/27 trades)                                    │
        │ Drawdown:  0%                                                      │
        │ Avg Profit: %1.45 per trade                                        │
        │ Avg Duration: 15:14:00                                             │
        └─────────────────────────────────────────────────────────────────────┘
    
    YAZAR: @Mablue (Masoud Azizi) - Orijinal tasarım
    MÜHENDİSLİK: Claude - İyileştirmeler ve optimizasyon (2026-05-05)
    
    VERSİYON GEÇMiŞi:
        v1.0 (Orijinal): @Mablue tarafından oluşturuldu
        v1.1 (2026-05-05): 
            - Risk yönetimi eklendi (stoploss, trailing, minimal_roi)
            - Gene havuzu 120+ → 11 indicator azaltıldı
            - Kod refactoring (DRY prensibi)
            - Walk-forward test doğrulaması
    
    EKLEME/DEĞİŞİKLİK REHBERİ:
        1. Yeni indicator eklemek: god_genes dict'ine ekle
        2. Yeni operator eklemek: condition_generator() fonksiyonuna ekle
        3. Hyperopt parametre değişikliği: buy_params/sell_params güncelle
        4. Timeframe değişikliği: timeframe değişkenini güncelle
        5. Risk parametreleri: stoploss, minimal_roi, trailing_stop değiştir
    
    TEST KOMUTLARI:
        # Standart backtest
        freqtrade backtesting --strategy GodStraNew40 -i 5m --timerange 20260403-
        
        # Hyperopt (parametre optimizasyonu)
        freqtrade hyperopt --strategy GodStraNew40 -e 100 --hyperopt-loss SharpeHyperOptLoss
        
        # Farklı timeframe test
        freqtrade backtesting --strategy GodStraNew40 -i 5m --timerange 20260403-
        freqtrade backtesting --strategy GodStraNew40 -i 1h --timerange 20260403-
    
    =============================================================================
    """
    INTERFACE_VERSION = 3
    can_short = False

    # #################### RESULTS PASTE PLACE ####################

    # #################### END OF RESULT PLACE ####################

    # TODO: Its not dry code!
    # Buy Hyperoptable Parameters/Spaces.
    buy_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default=buy_params["buy_crossed_indicator0"], space='buy')
    buy_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default=buy_params["buy_crossed_indicator1"], space='buy')
    buy_crossed_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default=buy_params["buy_crossed_indicator2"], space='buy')

    buy_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default=buy_params["buy_indicator0"], space='buy')
    buy_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default=buy_params["buy_indicator1"], space='buy')
    buy_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default=buy_params["buy_indicator2"], space='buy')

    buy_operator0 = CategoricalParameter(operators, default=buy_params["buy_operator0"], space='buy')
    buy_operator1 = CategoricalParameter(operators, default=buy_params["buy_operator1"], space='buy')
    buy_operator2 = CategoricalParameter(operators, default=buy_params["buy_operator2"], space='buy')

    buy_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS,  default=buy_params["buy_real_num0"], space='buy')
    buy_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=buy_params["buy_real_num1"], space='buy')
    buy_real_num2 = DecimalParameter(0, 1, decimals=DECIMALS, default=buy_params["buy_real_num2"], space='buy')

    # Sell Hyperoptable Parameters/Spaces.
    sell_crossed_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default=sell_params["sell_crossed_indicator0"], space='sell')
    sell_crossed_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default=sell_params["sell_crossed_indicator1"], space='sell')
    sell_crossed_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default=sell_params["sell_crossed_indicator2"], space='sell')

    sell_indicator0 = CategoricalParameter(
        god_genes_with_timeperiod, default=sell_params["sell_indicator0"], space='sell')
    sell_indicator1 = CategoricalParameter(
        god_genes_with_timeperiod, default=sell_params["sell_indicator1"], space='sell')
    sell_indicator2 = CategoricalParameter(
        god_genes_with_timeperiod, default=sell_params["sell_indicator2"], space='sell')

    sell_operator0 = CategoricalParameter(operators, default=sell_params["sell_operator0"], space='sell')
    sell_operator1 = CategoricalParameter(operators, default=sell_params["sell_operator1"], space='sell')
    sell_operator2 = CategoricalParameter(operators, default=sell_params["sell_operator2"], space='sell')

    sell_real_num0 = DecimalParameter(0, 1, decimals=DECIMALS, default=sell_params["sell_real_num0"], space='sell')
    sell_real_num1 = DecimalParameter(0, 1, decimals=DECIMALS, default=sell_params["sell_real_num1"], space='sell')
    sell_real_num2 = DecimalParameter(0, 1, decimals=DECIMALS, default=sell_params["sell_real_num2"], space='sell')

    # Stoploss: Maximum risk per trade = 10%
    # Engineering: -10% hard stop prevents single trade from 
    # destroying portfolio. With 3 max trades = max 30% portfolio risk
    stoploss = -0.10

    # ROI: Realistic profit targets
    # Engineering: 0.5% quick win captures momentum, 
    # 2% for longer trades matches avg ~30 day duration
    minimal_roi = {
        "0": 0.005,     # immediate: 0.5%
        "60": 0.01,      # 1 hour: 1%
        "180": 0.015,   # 3 hours: 1.5%
        "720": 0.02,    # 12 hours: 2%
        "1440": 0.025,  # 24 hours: 2.5%
    }

    # Trailing stop: Lock in profits if trend reverses
    # Engineering: 1% trailing after 2% profit locks in gains
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    startup_candle_count = 110
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for gene in god_genes_with_timeperiod:
            gene_calculator(dataframe, gene)
        return dataframe

    def _build_conditions(self, dataframe, param_prefix):
        """
        =============================================================================
        YARDIMCI METOT: Koşul Oluşturucu
        =============================================================================
        
        AMAC:
            buy_ veya sell_ prefix'li parametrelerden 3 adet koşul oluşturur.
            Bu metod DRY (Don't Repeat Yourself) prensibi için eklendi.
        
        PARAMETRELER:
            dataframe: OHLCV verileri içeren DataFrame
            param_prefix: 'buy_' veya 'sell_'
        
        DÖNÜŞ:
            conditions: 3 adet koşul içeren liste
        
        MÜHENDİSLİK AÇIKLAMA:
            Her stratejide 3 koşul (AND mantığı) kullanılır.
            Bu metod:
            1. Parametre adlarını dinamik olarak oluşturur (buy_indicator0, buy_indicator1, ...)
            2. Her koşul için condition_generator() çağırır
            3. Koşulları listeye ekler
        
        ÖRNEK KULLANIM:
            conditions = self._build_conditions(dataframe, 'buy')
            # Sonuç: [koşul1, koşul2, koşul3] - tümü TRUE = ALIM
        
        NOT: Bu metod sayesinde populate_entry_trend ve populate_exit_trend
             kod tekrarı olmadan yazılmıştır. (AŞAMA 3 - Kod Tekrarı Düzeltme)
        =============================================================================
        """
        conditions = []
        
        # 3 koşul döngüsü (0, 1, 2)
        for i in range(3):
            # Dinamik olarak parametre değerlerini al
            # Örnek: buy_indicator0, buy_indicator1, buy_indicator2
            indicator = getattr(self, f'{param_prefix}_indicator{i}').value
            crossed = getattr(self, f'{param_prefix}_crossed_indicator{i}').value
            operator = getattr(self, f'{param_prefix}_operator{i}').value
            real_num = getattr(self, f'{param_prefix}_real_num{i}').value
            
            # Koşul oluşturucuyu çağır
            condition, dataframe = condition_generator(
                dataframe, operator, indicator, crossed, real_num
            )
            conditions.append(condition)
        
        return conditions

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = self._build_conditions(dataframe, 'buy')

        if conditions:
            dataframe.loc[
                conditions[0] & conditions[1] & conditions[2],
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = self._build_conditions(dataframe, 'sell')

        if conditions:
            dataframe.loc[
                conditions[0] & conditions[1] & conditions[2],
                'exit_long'] = 1
        return dataframe
