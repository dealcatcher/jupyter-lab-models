# Thesis Outline: Discount Prediction & Price Forecasting in Retail Data

## Suggested Title Options:
- "Machine Learning Approaches for Retail Discount Prediction: A Two-Stage Framework"
- "Time Series Forecasting in Retail Pricing: Comparative Analysis of ARIMA and Hybrid Models"
- "Handling Class Imbalance in Retail Discount Forecasting: A Two-Stage Machine Learning Approach"

---

## Chapter 1: Introduction

### What to Write:
- **Background & Context**: Discuss the importance of pricing optimization in retail, the role of discounts in consumer behavior, and the business value of accurate discount forecasting
- **Problem Statement**: Explain the challenge of predicting discounts when 80% of records have zero discount (class imbalance problem)
- **Research Questions**: 
  - Can time series models effectively predict retail discounts?
  - How does a two-stage approach compare to traditional ARIMA?
  - What preprocessing steps are necessary for skewed retail pricing data?
- **Objectives**: 
  - Develop and compare ARIMA vs Two-Stage models
  - Address class imbalance through classification + regression pipeline
  - Evaluate model performance using RMSE%, residual analysis, and diagnostic checks
- **Scope & Limitations**: Focus on electronics/grocery pricing data, acknowledge data quality issues encountered
- **Thesis Structure**: Brief overview of remaining chapters

### Key Points from Your Documents:
- Reference the 80/20 split (6,500 zero-discount vs 1,400 discounted records)
- Mention the transition from Open Prices dataset due to missing electronics category
- Highlight the practical business need for discount prediction

---

## Chapter 2: Literature Review

### What to Write:
- **Time Series Forecasting**: Review ARIMA, SARIMA foundations and applications in retail
- **Class Imbalance Solutions**: Discuss techniques like class weighting, threshold adjustment, SMOTE
- **Two-Stage/Hurdle Models**: Cover zero-inflated models, classification-regression pipelines
- **Retail Pricing Analytics**: Previous work on price optimization, discount patterns, promotional forecasting
- **Diagnostic Testing**: Literature on residual analysis, Ljung-Box, Shapiro-Wilk in forecasting validation
- **Gaps Identified**: Limited work on two-stage approaches for retail discount specifically

### Key Sources to Cite:
- Box-Jenkins methodology for ARIMA
- Research on imbalanced classification (churn, fraud detection parallels)
- Retail analytics case studies

---

## Chapter 3: Methodology

### What to Write:

#### 3.1 Dataset Description
- **Original Dataset**: Open Prices from Hugging Face (235K rows × 27 columns)
- **Data Quality Issues**: 
  - Missing values in product_name, category_tag, origins_tags
  - All-null columns removed (location_website_url, location_source, etc.)
  - Price range skewness (0 to 1,000,000)
  - Electronics category absent → dataset transition decision
- **Final Dataset Characteristics**: Class distribution, feature descriptions

#### 3.2 Data Preprocessing
- **Column Selection**: Removing entirely null columns
- **Feature Engineering**: 
  - Created actual_price column (price before discount)
  - Renamed price → price_after_discount
  - Added is_discounted binary flag
- **Handling Skewness**: Price segmentation (0-1,000 vs 1,000-1,000,000)
- **Train-Test Split**: Time-based or random split methodology

#### 3.3 Model 1: ARIMA
- **Theoretical Foundation**: ARIMA(p,d,q) components
- **Parameter Selection Process**:
  - ACF analysis → MA(q) term identification (Lag 1, Lag 4 spikes)
  - PACF analysis → AR(p) term identification (Lag 1, Lag 2 dominant)
  - ADF Test → Stationarity confirmation (d=0, p-value < 0.05)
- **Final Parameters**: ARIMA(2,0,0) as determined by RMSE minimization
- **Implementation Details**: Library used, hyperparameter search space

#### 3.4 Model 2: SARIMA
- **Seasonal Extension**: Adding seasonal components (P,D,Q,s)
- **Parameter Selection**: (2,0,2,0,1,5,12) - explain each component
- **Rationale**: Capturing weekly/monthly discount patterns

#### 3.5 Model 3: Two-Stage Framework
- **Stage 1 - Classification**:
  - Algorithm: Logistic Regression with Gradient Boosting
  - Problem: Binary classification (is_discounted = 0 or 1)
  - Class Imbalance Fixes:
    - class_weight='balanced' (inverse frequency weighting)
    - Threshold adjustment: 0.5 → 0.3 (increase recall)
  - Hyperparameters: n_estimators, max_depth, learning_rate tuning
  
- **Stage 2 - Regression**:
  - Algorithm: Gradient Boosting Regressor
  - Input: Only records predicted as discounted (is_discounted=1)
  - Target: Discount percentage/amount
  - Features: Category, temporal features, price features
  
- **Pipeline Integration**: How Stage 1 output feeds Stage 2

#### 3.6 Evaluation Metrics
- **RMSE%**: (RMSE / mean_actual) × 100 - normalized error interpretation
  - <10%: Excellent, 10-20%: Acceptable, 20-30%: Weak, >30%: Poor
- **MAE**: Mean Absolute Error - average magnitude of errors
- **Residual Mean**: Bias detection (should be ≈0)
- **Ljung-Box Test**: Residual autocorrelation (p≥0.05 = pass)
- **Shapiro-Wilk Test**: Residual normality (p≥0.05 = pass)
- **Skewness**: Distribution symmetry (−1 to +1 acceptable)

#### 3.7 Diagnostic Framework
- Present the 6-check evaluation system from Testing.docx.pdf
- Explain why each check matters for production reliability

---

## Chapter 4: Results

### What to Write:

#### 4.1 ARIMA Results
- **Parameters**: ARIMA(2,0,0)
- **Performance**:
  - RMSE%: 343.02% (FAIL - extremely poor)
  - Residual Mean: 0.5513 (slight under-prediction bias)
  - Ljung-Box p-value: 0.0 (FAIL - autocorrelation remains)
  - Shapiro-Wilk p-value: 0.0 (FAIL - non-normal residuals)
  - MAE: 23.77%
  - Skewness: 3.99 (FAIL - heavily right-skewed)
- **Interpretation**: ARIMA struggles with zero-inflated data, spike-like discount patterns

#### 4.2 SARIMA Results
- **Parameters**: (2,0,2,0,1,5,12)
- **Performance**:
  - RMSE%: 105.52% (FAIL - still poor but better than ARIMA)
  - Residual Mean: 23.76 (significant bias)
  - Ljung-Box p-value: 0.32 (PASS - residuals random)
  - Shapiro-Wilk p-value: 0.00 (FAIL - non-normal)
  - MAE: 23.1%
  - Skewness: 0.72 (PASS - acceptable symmetry)
- **Interpretation**: Seasonal components help but don't solve fundamental zero-inflation problem

#### 4.3 Two-Stage Model Results
- **Performance**:
  - RMSE%: 45% (WARN - weak but usable)
  - Residual Mean: -0.2 (PASS - minimal bias)
  - Ljung-Box p-value: 0.0 (FAIL - some pattern remains)
  - Shapiro-Wilk p-value: 0.0 (FAIL - non-normal)
  - MAE: 19% (PASS - best of three models)
  - Skewness: 0.88 (PASS - acceptable)
- **Interpretation**: Best overall performance, handles class imbalance effectively

#### 4.4 Comparative Analysis Table
| Metric | ARIMA | SARIMA | Two-Stage | Ideal |
|--------|-------|--------|-----------|-------|
| RMSE% | 343.02% ❌ | 105.52% ❌ | 45% ⚠️ | <20% |
| Residual Mean | 0.55 | 23.76 | -0.2 | ≈0 |
| Ljung-Box p | 0.0 ❌ | 0.32 ✅ | 0.0 ❌ | ≥0.05 |
| Shapiro-Wilk p | 0.0 ❌ | 0.0 ❌ | 0.0 ❌ | ≥0.05 |
| MAE | 23.77% | 23.1% | 19% ✅ | <20% |
| Skewness | 3.99 ❌ | 0.72 ✅ | 0.88 ✅ | −1 to +1 |

#### 4.5 Visualizations to Include
- ACF/PACF plots for ARIMA parameter selection
- ADF test results showing stationarity
- Class distribution bar chart (80/20 split)
- Residual histograms for all three models
- Actual vs Predicted comparison plots
- Heatmap of correlation between features

---

## Chapter 5: Discussion

### What to Write:

#### 5.1 Why ARIMA Failed
- Assumption of continuous time series violated by 80% zeros
- Cannot handle structural breaks (discount vs no-discount regimes)
- Linear model inadequate for promotional spike patterns
- High RMSE (9.79 on 0-80% scale) confirms poor fit

#### 5.2 Two-Stage Model Advantages
- **Decoupling**: Separates "whether" from "how much" questions
- **Class Imbalance**: Directly addressed through weighted classification
- **Threshold Tuning**: Lowering from 0.5→0.3 improved recall (business-appropriate tradeoff)
- **Specialized Models**: Each stage optimized for its specific task
- **Best MAE**: 19% vs 23%+ for time series models

#### 5.3 Remaining Challenges
- **Ljung-Box Failure**: Even two-stage model has residual autocorrelation
  - Possible causes: Unmodeled seasonality, external factors (holidays, promotions)
  - Future work: Add exogenous variables (marketing spend, competitor prices)
- **Shapiro-Wilk Failure**: All models show non-normal residuals
  - Indicates multiplicative seasonality or heteroscedasticity
  - Potential fix: Log transformation, Box-Cox
- **Data Quality Impact**: Missing product_name limited feature engineering
  - Category-level analysis impossible without category_tag
  - Origin-based patterns unavailable

#### 5.4 Business Implications
- **False Negatives Cost**: Missing actual discounts worse than false alarms
  - Justifies lower classification threshold (0.3)
  - Aligns with business priority: capture promotion opportunities
- **Forecasting Accuracy**: 45% RMSE% means predictions have uncertainty
  - Suitable for strategic planning, not real-time pricing
  - Recommend human oversight for critical decisions

#### 5.5 Comparison to Literature
- Align findings with existing research on zero-inflated models
- Discuss how results support/differ from prior retail forecasting studies
- Position two-stage approach within broader hurdle model literature

---

## Chapter 6: Conclusion & Future Work

### What to Write:

#### 6.1 Summary of Contributions
- Demonstrated inadequacy of standard ARIMA for zero-inflated discount data
- Developed and validated two-stage classification-regression framework
- Established diagnostic evaluation framework (6-check system)
- Documented data quality challenges in open retail datasets

#### 6.2 Key Findings
1. **Class imbalance is critical**: 80/20 split requires specialized handling
2. **Two-stage outperforms**: 45% vs 105-343% RMSE% improvement
3. **Diagnostics matter**: Low RMSE alone insufficient; need residual analysis
4. **Data quality limits models**: Missing features constrain accuracy

#### 6.3 Limitations
- Single dataset (electronics/grocery focus)
- Limited external variables (no marketing, competitor, macroeconomic data)
- Short time horizon (may not capture long-term trends)
- Non-normal residuals suggest model misspecification

#### 6.4 Future Research Directions
1. **Feature Enhancement**:
   - Incorporate marketing calendar, holiday indicators
   - Add competitor pricing data
   - Include product lifecycle stage
   
2. **Model Improvements**:
   - Try XGBoost/LightGBM for both stages
   - Implement deep learning (LSTM for temporal patterns)
   - Bayesian structural time series for uncertainty quantification
   
3. **Preprocessing Alternatives**:
   - SMOTE for synthetic minority samples
   - Zero-inflated Poisson/Negative Binomial models
   - Quantile regression for prediction intervals
   
4. **Deployment Considerations**:
   - Real-time inference pipeline
   - Model drift monitoring
   - A/B testing framework for forecast impact

#### 6.5 Practical Recommendations
- Use two-stage model for discount prediction tasks with class imbalance
- Always perform full diagnostic checks, not just RMSE
- Invest in data quality (product categorization, feature completeness)
- Combine ML forecasts with domain expert judgment

---

## Chapter 7: References

### What to Include:
- ARIMA/SARIMA foundational papers (Box & Jenkins)
- Class imbalance literature (Chawla et al. on SMOTE)
- Retail pricing analytics case studies
- Diagnostic testing references (Ljung-Box, Shapiro-Wilk original papers)
- Python libraries documentation (statsmodels, scikit-learn, xgboost)

---

## Appendices

### Appendix A: Code Repository
- Link to GitHub with full implementation
- Jupyter notebooks for reproducibility
- Requirements.txt with package versions

### Appendix B: Additional Tables
- Full hyperparameter search grids
- Complete confusion matrices for classifier
- Feature importance rankings

### Appendix C: Extended Diagnostics
- Q-Q plots for residual normality
- Autocorrelation plots for all lags
- Time series decomposition plots

### Appendix D: Data Dictionary
- Complete column descriptions from final dataset
- Transformation formulas applied
- Missing value statistics

---

## Writing Tips Based on Your Documents:

### From Model_Performance_Summary:
- Use the exact parameter values and RMSE numbers you calculated
- Include the ACF/PACF interpretation logic (Lag 1, Lag 2, Lag 4 observations)
- Explain the ADF test result clearly (stationary → d=0)

### From Testing.docx.pdf:
- Frame the 6-check diagnostic system as your evaluation framework
- Use the pass/fail thresholds (<10% excellent, >30% poor for RMSE%)
- Include the analogy about coin flips for Ljung-Box explanation

### From Data_Documentation:
- Be honest about data quality issues (null columns, missing electronics)
- Document the decision-making process for dataset transition
- Show the price segmentation strategy (0-1000 vs 1000-1000000)

---

## Suggested Word Count Distribution:
- Chapter 1 (Intro): 3,000-4,000 words
- Chapter 2 (Literature): 8,000-10,000 words
- Chapter 3 (Methodology): 10,000-12,000 words
- Chapter 4 (Results): 8,000-10,000 words
- Chapter 5 (Discussion): 6,000-8,000 words
- Chapter 6 (Conclusion): 3,000-4,000 words
- **Total**: ~40,000-50,000 words (typical Master's thesis)

---

## Next Steps:
1. Start with Chapter 3 (Methodology) since you have all the technical details
2. Write Chapter 4 (Results) with your actual numbers and visualizations
3. Draft Chapter 1 (Introduction) to frame what you've already done
4. Fill in Chapter 2 (Literature) with supporting research
5. Complete Discussion and Conclusion last

Good luck with your thesis! 📚✨
