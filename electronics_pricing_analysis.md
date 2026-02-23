# Electronics Products Pricing Data — Full Analysis Report

> **Dataset:** Electronics Products Pricing (2017–2018)  
> **Goal:** Understand discount behavior and build a forecasting model  
> **Tools:** Python (pandas, statsmodels, sklearn)

---

## 1. Autocorrelation Analysis

**Chart:** Autocorrelation Plot (ACF)

### Findings
- Strong positive autocorrelation at **early lags (1–5)**, decaying slowly toward zero
- The **slow, gradual decay** (rather than a sharp cutoff) is characteristic of an **AR (AutoRegressive) process**
- The series shows **persistence and long-term memory** — past discount values strongly influence near-future values
- No sharp seasonal spikes visible in the ACF, but the gradual pattern suggests a **non-stationary trend component**

### Implications
- The data is likely **non-stationary** → differencing will be needed before modeling
- An **ARIMA or SARIMA model** is appropriate as a starting point
- Short-term forecasts (1–7 days ahead) will benefit from lag features

---

## 2. 📅 Discount Patterns by Day of Week (2017 vs 2018)

**Chart:** Average Discount Percent by Day of Week — Bar Chart

### Findings

| Day       | 2017 Avg Discount % | 2018 Avg Discount % | Notable Change     |
|-----------|---------------------|---------------------|--------------------|
| Monday    | ~3.5%               | ~7.0%               | ⬆️ Big increase     |
| Tuesday   | ~2.5%               | ~2.0%               | ➡️ Stable / slight drop |
| Wednesday | ~3.5%               | ~1.2%               | ⬇️ Notable decrease |
| Thursday  | ~7.3%               | ~11.0%              | ⬆️ Largest discounts |
| Friday    | ~7.0%               | ~5.3%               | ⬇️ Slight decrease  |
| Saturday  | ~1.5%               | ~0.4%               | ⬇️ Lowest overall   |
| Sunday    | ~5.0%               | ~1.6%               | ⬇️ Large decrease   |

### Key Takeaways
- **Thursday is the prime discount day** in both years, and got even more aggressive in 2018 (~11%)
- **Saturday consistently has the lowest discounts** — likely full-price positioning
- Wide **confidence intervals on Thursday and Monday** indicate high variability — averages must be interpreted with caution
- Discount strategy **shifted significantly** between 2017 and 2018, making year-over-year comparisons tricky

---

## 3. 📆 Seasonal Discount Patterns by Month (2017 vs 2018)

**Chart:** Discount Seasonality using Date Seen — Line Chart

### Findings

| Period          | 2017 Pattern                          | 2018 Pattern                          |
|-----------------|---------------------------------------|---------------------------------------|
| Jan–Mar (Q1)    | Near 0% discounts                     | Strong peak ~11% in March             |
| Apr–Jun (Q2)    | Near 0% discounts                     | Sharp collapse to ~0% by June         |
| Jul–Sep (Q3)    | Rising gradually from ~1.5% to ~4.5%  | No data / flat                        |
| Oct–Dec (Q4)    | Sharp rise to ~10% in November (holiday season) | No data / flat             |

### Key Takeaways
- **2017 and 2018 follow nearly opposite seasonal patterns** — this is the most important and challenging finding
- 2017 follows a **classic holiday-season discount surge** (Q4 peak = Black Friday / Christmas behavior)
- 2018 shows a **Q1 peak** instead — possibly a new promotional strategy or data coverage difference
- The **unstable seasonality between years** makes traditional seasonal models (e.g., simple SARIMA) risky
- **Prophet with changepoint detection** is strongly recommended to handle this instability

---

## 4. 🔥 Correlation Heatmap

**Chart:** Electronics Products Pricing Data Correlation Heatmap

### Correlation Matrix Summary

| Variable Pair                          | Correlation | Interpretation                        |
|----------------------------------------|-------------|---------------------------------------|
| prices.amountMax ↔ prices.amountMin    | **0.97**    | Near-perfect linear relationship      |
| price_difference ↔ discount_percent   | **0.62**    | Moderate positive — bigger gaps = bigger discounts |
| prices.amountMax ↔ price_difference   | **0.53**    | Higher-priced items have wider spreads |
| prices.amountMin ↔ price_difference   | **0.31**    | Weaker relationship                   |
| prices.amountMax ↔ discount_percent   | **0.25**    | Weak — price level barely predicts discount |
| date_year ↔ all variables             | **~0.03**   | No linear time trend in pricing       |

### Key Takeaways
- `prices.amountMax` and `prices.amountMin` are nearly redundant — only one is needed in models (avoid multicollinearity)
- `price_difference` is the **best price-based predictor of discount %** (r = 0.62)
- **Year alone does not predict pricing** — confirming that changes are non-linear and seasonal, not a simple trend
- `discount_percent` has **low correlation with raw prices** — discount behavior is driven by timing and strategy, not just price level

---

## 5. 🔵 Scatterplot Analysis (Prices & Discounts over Time)

**Chart:** 4-Panel Scatterplot — Min/Max Prices, Discount % vs Date Added/Seen/Updated

### Panel 1: Min vs Max Prices
- Strong **linear relationship** confirms the heatmap finding (r = 0.97)
- Several **outliers above $5,000** — likely premium/enterprise electronics
- Majority of products are **clustered below $2,000**

### Panel 2: Discount % vs Date Added
- Significant discounting activity **emerged and intensified from 2017 onward**
- Early listings (2014–2016) show sparse, sporadic discounts
- A **two-tier structure** is clear: many products permanently at 0% discount, a separate group with active discounting

### Panel 3: Discount % vs Date Seen
- Most discount activity **concentrated in late 2017 – early 2018** window
- Discounts range widely (0–80%), indicating **no standardized discount policy**

### Panel 4: Discount % vs Date Updated
- Similar pattern to Date Seen — discount activity is **recent and growing**
- Outliers at 75–80% discount are present but rare

### Key Takeaways
- The **two-tier market structure** (discounted vs. full-price products) is a critical modeling challenge
- Recommend a **two-stage model**: Stage 1 = classify "will this product be discounted?", Stage 2 = predict "how much?"
- The catalog of actively discounted products has been **expanding over time**

---

## 6. 📈 Lag-1 Relationship

**Chart:** Lag-1 Scatter Plot — Current Discount (t) vs Previous Discount (t-1)

### Findings
- **No strong linear structure** visible in the lag-1 scatter
- Points are widely dispersed with no clear trend line
- This seems to contradict the ACF results, but is explained by **aggregation level** — the lag-1 scatter is at product level while ACF is at series level

### Key Takeaways
- Discount changes at the **individual product level** are relatively unpredictable from one period to the next
- Predictability comes from **aggregate time series patterns** (ACF), not product-level lag-1 relationships
- Models should focus on **time-series level forecasting** rather than individual product-level lag regression

---

## 7. 🗺️ Recommended Next Steps — Modeling Roadmap

### Step 1 — Data Preprocessing & Feature Engineering
- Handle two-tier discount structure (classify → regress)
- Create: `day_of_week`, `month`, `is_thursday`, `lag_1`, `lag_7`, `rolling_4w_avg`, `price_difference`
- Train on 2017, test on 2018 (time-respecting split)

### Step 2 — Baseline Model
- Seasonal Naive: predict = same day last week
- Metrics: RMSE, MAE

### Step 3 — Forecasting Future Discount %
- **Prophet** (recommended): handles unstable seasonality + changepoints automatically
- **SARIMA**: good backup for stable periods, use `order=(2,1,1)`, `seasonal_order=(1,1,1,7)`

### Step 4 — Best Day to Apply Discounts
- **Gradient Boosting Regressor** with features: `day_of_week`, `month`, `lag_1`, `lag_7`, `price_difference`
- Validate with **SHAP values** to confirm Thursday dominance

### Step 5 — Seasonal Planning
- Seasonal decomposition using `statsmodels.tsa.seasonal.seasonal_decompose`
- Prophet changepoint analysis to find where seasonality shifts between years

### Model Comparison Summary

| Model              | Best For                    | Complexity |
|--------------------|-----------------------------|------------|
| Seasonal Naive     | Baseline benchmark          | Low        |
| SARIMA             | Stable seasonality          | Medium     |
| **Prophet**        | Changing seasonality        | Low ✅ Start here |
| **Gradient Boosting** | Feature-rich, day-of-week | Medium ✅ Start here |
| LSTM               | Long-term complex patterns  | High       |

---

## 8. 📌 Summary of Critical Findings

| # | Finding | Impact |
|---|---------|--------|
| 1 | Thursday = highest discount day (both years) | Use for promotional scheduling |
| 2 | Saturday = lowest discount day consistently | Full-price positioning day |
| 3 | 2017 vs 2018 seasonality is nearly opposite | Unstable seasonality — use Prophet |
| 4 | price_difference is the best price predictor of discount | Include as key feature |
| 5 | Two-tier market: 0% vs actively discounted products | Two-stage model required |
| 6 | Discounting activity intensified post-2016 | Expanding discount strategy over time |
| 7 | ACF shows slow decay → AR process + non-stationary | Differencing needed before SARIMA |
| 8 | date_year has near-zero correlation with prices | No simple linear time trend |

---

*Report generated from visual analysis of 6 charts: ACF plot, Day-of-Week bar chart, Seasonality line chart, Correlation heatmap, 4-panel scatterplot, and Lag-1 scatter plot.*
