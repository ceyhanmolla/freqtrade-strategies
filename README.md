# Freqtrade Strategies

Backtested and optimized Freqtrade strategy collection.

## Environment

| Parameter | Value |
|-----------|-------|
| Freqtrade Version | **2026.4** (Docker, stable) |
| Exchange | Binance Spot |
| Start Balance | 100 USDT |
| Stake | Unlimited (balance split equally) |
| Max Open Trades | 5 |

---

## 3-Month Backtest Results (2026-02-08 → 2026-05-13)

| # | Strategy | TF | Profit % | Trades | Win Rate | DD | PF (Est.) |
|---|----------|-----|----------|--------|----------|-----|-----------|
| 1 | **GeneticEngineV1** | 5m | **+42.56%** | 183 | 98.9% | 1.65% | **90+** |
| 2 | **FSampleStrategy** | 1h | +15.72% | 18 | 55.6% | 3.32% | ~1.25 |
| 3 | **TrendFollowingStrategyV2** | 5m | +8.39% | 53 | 75.5% | 9.87% | ~3.08 |
| 4 | **AdaptiveMomentum** | 15m | +3.83% | 154 | 89.6% | 10.24% | ~8.63 |
| 5 | **EwoMomentumV1** | 5m | +3.76% | 171 | 77.2% | 13.97% | ~3.38 |
| 6 | **ContraRatingsStrategy** | 1m | +3.23% | 89 | 82.0% | 3.41% | ~4.56 |
| 7 | **EVX_Tactical** | 15m | -16.70% | 624 | 69.9% | 22.85% | <1 (neg) |

**PF Notu:** Profit Factor = Toplam Kar / Toplam Zarar. Tahmini değerler win/loss oranından hesaplandı. Gerçek PF için trade-by-trade veri gereklidir.

---

## Active Strategy: GeneticEngineV1

### Live Configuration
```
Strategy: GeneticEngineV1
Timeframe: 5m
Max Open Trades: 5
Freqtrade: 2024.9 (Docker)
```

### Transition Detection Fix
**Problem:** Multiple entries on same signal
**Solution:** Added `combined & (combined.shift(1) == False)` logic to ensure only one entry per signal transition (False→True)

### Backtest Results
```
Profit: +42.56% | Trades: 183 | Win Rate: 98.9% | DD: 1.65% | PF: 90+
```

---

## Strategy Details

### 1. GeneticEngineV1 (5m) — Active (Live Testing)
```
Profit: +42.56% | Trades: 183 | Win Rate: 98.9% | DD: 1.65% | PF: 90+
```
- **Logic**: Genetic algorithm with 3-condition AND logic for entry/exit
- **Gene Pool**: SMA, EMA, DEMA, TEMA, KAMA, RSI, MACD, STOCH, CCI, BBANDS, ATR
- **Key Fix**: Transition detection — sadece False→True geçişinde sinyal
- **Entry**: 3 koşul AND ile birleştiğinde (transition kontrolü ile)
- **Exit**: 3 koşul AND ile birleştiğinde (transition kontrolü ile)
- **Status**: Remote sunucuda canlı test aşamasında

### 2. ContraRatingsStrategy (1m) — Best Profit Factor
```
Profit: +14.09% | Trades: 135 | Win Rate: 86.7% | DD: 4.14% | PF: 2.95
```
- **Logic**: Multi-timeframe contrarian — bullish ratings from 5m & 1m timeframes for entry, bearish for exit
- **Entry**: 5m + 1m bullish >= 3 (strong buy signal)
- **Exit**: minimal_roi kademeli — 4%, 2.5%, 1.5%, 0.8% (0-120-360-720 dk)
- **Custom Exit**: 48h timeout (max_hold: 172800)
- **Strength**: Best profit factor (2.95), low drawdown (4.14%), high win rate

### 3. EVX_Tactical (1h)
```
Profit: +27.40% | Trades: 649 | Win Rate: 72.9% | DD: 11.66%
```
- **Logic**: Excess Volume Index — bid/ask volume imbalance via `(close-low)/(high-low)` ratio

### 4. FSampleStrategy (5m) — Low-Trade Specialist
```
Profit: +13.59% | Trades: 61 | Win Rate: 36.1% | DD: 16.34%
```
- **Note**: 1-month backtest showed +41.15% but 3-month reveals higher drawdown risk

### 5. EwoMomentumV1 (5m)
```
Profit: +14.47% | Trades: 53 | Win Rate: 86.8% | DD: 4.15%
```
- **Logic**: EWO (Elliot Wave Oscillator) = SMA(50) - SMA(200)

### 6. AdaptiveMomentum (15m) — Lowest Drawdown
```
Profit: +13.67% | Trades: 83 | Win Rate: 95.2% | DD: 1.03%
```
- **Logic**: Multi-signal entry (ADX+RSI, EMA+MACD, Oversold RSI) + DCA

### 7. TrendFollowingStrategyV2 (5m)
```
Profit: +8.26% | Trades: 53 | Win Rate: 79.2% | DD: 11.08%
```
- **Logic**: EMA(20) crossover + OBV + 1h EMA(50) filter

---

## Engineering Notes

- **1-month backtests are misleading** — Always test across multiple time ranges.
- **GeneticEngineV1** offers the best risk-adjusted performance with highest win rate (98.9%) and lowest drawdown (1.65%).
- **Transition Detection**: Kritik düzeltme — her sinyal için sadece 1 entry. `combined & (combined.shift(1) == False)` formülü kullanılıyor.
- All strategies use `INTERFACE_VERSION = 3` (Freqtrade V3 format).
- **Freqtrade 2024.9** — Docker'da çalışıyor, eski Python 2020.8 local'de çalışmıyordu.

---

## Usage

```bash
# GeneticEngineV1 (5m - aktif canlı)
freqtrade backtesting --strategy GeneticEngineV1 -i 5m

# ContraRatingsStrategy (1m)
freqtrade backtesting --strategy ContraRatingsStrategy -i 1m

# EVX_Tactical
freqtrade backtesting --strategy EVX_Tactical_Strategy -i 1h

# EwoMomentumV1
freqtrade backtesting --strategy EwoMomentumV1 -i 5m

# AdaptiveMomentum
freqtrade backtesting --strategy AdaptiveMomentum -i 15m
```

---

**Risk Warning**: Crypto trading carries high risk. Backtest results do not guarantee future performance. Always start with dry-run mode.