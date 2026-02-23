# Electronics Products Pricing — Full Analysis Report

> **Dataset:** Electronics Products Pricing Data (2017–2018)  
> **Objective:** Understand discount behavior and build a foundation for forecasting future discounts

---

## 1. Stationarity Test — ADF (Augmented Dickey-Fuller)

| Metric | Value |
|---|---|
| **Test** | Augmented Dickey-Fuller (ADF) |
| **p-value** | `2.2203037811309657e-11` |
| **Result** | **Stationary** |
| **Interpretation** | p-value << 0.05, so we **reject the null hypothesis** of a unit root. The discount percent time series is stationary — it does not have a persistent upward/downward trend over time. This is good news for time series modeling. |

---

## 2. Autocorrelation Analysis (ACF Plot)

| Observation | Detail |
|---|---|
| **Lag-0** | Perfect autocorrelation = 1.0 (expected) |
| **Lag-1 to Lag-3** | Strong positive autocorrelation (~0.55 → ~0.38) |
| **Lag-4 to Lag-20** | Gradual decay toward zero |
| **Lag-20+** | Near-zero, within confidence bands |
| **Pattern Type** | Slow exponential decay → suggests **AR (AutoRegressive) process** |
| **Implication** | Recent discount values significantly influence the next period's discount. Short-term memory is strong. |

**Modeling Recommendation:** AR(2) or SARIMA model is appropriate given the slow decay pattern.

---

## 3. Day-of-Week Discount Patterns (2017 vs 2018)

| Day | Avg Discount 2017 | Avg Discount 2018 | Trend |
|---|---|---|---|
| **Monday** | ~3.5% | ~7.0% | ⬆️ Increased |
| **Tuesday** | ~2.5% | ~2.0% | ➡️ Stable |
| **Wednesday** | ~3.5% | ~1.2% | ⬇️ Decreased |
| **Thursday** | ~7.3% | ~11.0% | ⬆️ **Highest — Peak Discount Day** |
| **Friday** | ~7.0% | ~5.3% | ⬇️ Slight decrease |
| **Saturday** | ~1.5% | ~0.4% | ⬇️ Lowest discount day |
| **Sunday** | ~5.0% | ~1.6% | ⬇️ Decreased |

**Key Findings:**
- **Thursday** is consistently the highest discount day across both years — likely a strategic promotional window
- **Saturday** has the lowest discounts — possibly reduced promotional activity on weekends
- ⚠️ High confidence intervals on Thursday and Monday suggest **high variability** — averages should be interpreted with caution

---

## 4. Seasonal Discount Patterns (Monthly, 2017 vs 2018)

| Period | 2017 Pattern | 2018 Pattern |
|---|---|---|
| **Jan–May (Months 1–5)** | Near zero discounts | Peak discounts (~5% → 11%) |
| **Jun (Month 6)** | Near zero | Collapse to ~0% |
| **Jul–Sep (Months 7–9)** | Gradual rise | No data / minimal |
| **Oct–Dec (Months 10–12)** | **Peak ~10%** (holiday season) | No data |

**Key Findings:**
- **The two years have nearly opposite seasonal patterns** — 2018 peaks early (Q1), 2017 peaks late (Q4)
- 2017 shows classic **holiday-season pricing** behavior with Q4 discounts rising sharply
- 2018's early peak (March) may reflect a **strategy shift** or data collection difference
- ⚠️ This instability means simple seasonal models will struggle — **changepoint-aware models** (e.g., Prophet) are needed

---




## 5. Scatterplot Analysis (Prices & Discounts Over Time)

### Min vs Max Price
- Strong linear relationship confirmed visually
- Notable **outliers above $5,000** — ultra-premium products behave differently
- Consider **segmenting** high-price products for separate modeling

### Discount % Over Time (dateAdded, dateSeen, dateUpdated)
- Discounting activity **became denser from 2017 onward** — more products with discounts
- Large cluster of **0% discount items persist throughout** — confirms a two-tier market
- Discount % ranges up to **~80%** — significant outliers present

---

## 6. Lag-1 Relationship

| Observation | Detail |
|---|---|
| **Structure** | No strong linear pattern between t and t-1 |
| **Spread** | Points scattered widely across the plot |
| **Implication** | Discount at the previous period is **not a strong standalone predictor** at this aggregation level |
| **Note** | This contrasts with the strong ACF at lag-1 — suggesting the relationship exists but is **non-linear** or requires additional features to surface |

---

## 7. Next Steps (Modeling Roadmap)

### Phase 1 — Data Preparation
- [ ] Engineer time features: `day_of_week`, `month`, `week_of_year`, `is_thursday`
- [ ] Create lag features: `lag_1`, `lag_7`, `rolling_4w_avg`
- [ ] Handle 0% discounts — build a **two-stage model** (classify → then regress)
- [ ] Train/test split: **2017 = train, 2018 = test**

### Phase 2 — Baseline
- [ ] Seasonal naive forecast (predict = same day last week)
- [ ] Evaluate with **RMSE** and **MAE**

### Phase 3 — Core Models
- [ ] **Prophet** — handles unstable seasonality and changepoints automatically Recommended
- [ ] **SARIMA(2,0,1)(1,1,1,7)** — leverages AR pattern found in ACF
- [ ] **Gradient Boosting** — feature-rich model for day-of-week optimization

### Phase 4 — Interpretation
- [ ] **SHAP values** — explain feature importance, validate Thursday finding
- [ ] **Seasonal decomposition** — formally separate trend, seasonality, residuals

### Model Comparison Summary

| Model | Strengths | Use For |
|---|---|---|
| Prophet | Unstable seasonality, changepoints | Seasonal planning |
| SARIMA | AR process, stationary data | Short-term forecasting |
| Gradient Boosting | Non-linear, feature-rich | Day-of-week optimization |
| LSTM | Long-term memory | Only if above models fall short |

---

## 9. Key Takeaways Summary

| # | Finding |
|---|---|
| 1 | Time series is **stationary** (ADF p = 2.22e-11) — safe to model without differencing |
| 2 | Strong **short-term autocorrelation** — recent discounts predict near-future discounts |
| 3 | **Thursday** is the prime discount day; Saturday is the weakest |
| 4 | Seasonal patterns are **unstable between years** — use changepoint-aware models |
| 5 | `price_difference` is the best pricing predictor of discount % (r=0.62) |
| 6 | **Two-tier market** exists: many products never discounted; model them separately |
| 7 | Discounting activity **intensified post-2016** — possible strategy shift |
| 8 | ⚠️ High-price outliers (>$5K) may need **separate segmented models** |

---

*Report generated from visual and statistical analysis of Electronics Products Pricing Data (2017–2018)*
