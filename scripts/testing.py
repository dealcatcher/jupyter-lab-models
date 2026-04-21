"""
=============================================================
  Model Performance Checker
  Supports: SARIMA and Two-Stage (Classifier + Regressor)

  Usage:
    # SARIMA
    check_model_performance(model_fit, train, test, forecast, best_params)

    # Two-Stage
    check_two_stage_performance(
        clf, regressor, train, test,
        final_forecast, zero_pred, zero_pred_proba,
        sarima_forecast, best_params
    )
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import shapiro, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from mlflow import log_metric, log_param
import mlflow
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

mlflow.set_tracking_uri("http://86.50.20.250/")

# ═══════════════════════════════════════════════════════════════
#  SHARED CHECKS  (used by both SARIMA and Two-Stage)
# ═══════════════════════════════════════════════════════════════

def check_rmse_percent(rmse, actual, threshold=20.0, nonzero_only=False):
    """RMSE as % of actual mean."""
    base   = actual[actual > 0].mean() if nonzero_only else actual.mean()
    label  = "non-zero mean" if nonzero_only else "actual mean"
    pct    = (rmse / base) * 100 if base != 0 else float("inf")
    passed = pct <= threshold
    return {
        "check"     : f"RMSE % (vs {label})",
        "value"     : f"{pct:.2f}%",
        "threshold" : f"<= {threshold}%",
        "passed"    : passed,
        "advice"    : "✅ RMSE % is acceptable." if passed
                      else f"❌ RMSE % too high ({pct:.1f}%) → try log transform or better features.",
    }


def check_residual_mean(residuals, tolerance=0.05):
    """Residual mean should be close to 0."""
    mean   = residuals.mean()
    passed = abs(mean) <= tolerance
    return {
        "check"     : "Residual Mean (Bias)",
        "value"     : f"{mean:.4f}",
        "threshold" : f"|mean| <= {tolerance}",
        "passed"    : passed,
        "advice"    : "✅ No systematic bias in residuals." if passed
                      else f"❌ Bias detected (mean={mean:.4f}) → model consistently over/under-predicts.",
    }


def check_ljung_box(residuals, lags=10, alpha=0.05):
    """Ljung-Box: residuals should NOT be autocorrelated."""
    lb     = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    pvalue = lb["lb_pvalue"].values[0]
    passed = pvalue >= alpha
    return {
        "check"     : "Ljung-Box (Autocorrelation)",
        "value"     : f"p = {pvalue:.4f}",
        "threshold" : f">= {alpha}",
        "passed"    : passed,
        "advice"    : "✅ Residuals are not autocorrelated." if passed
                      else "❌ Residuals autocorrelated → increase p/q or add differencing.",
    }


def check_shapiro(residuals, alpha=0.05):
    """Shapiro-Wilk normality test on residuals."""
    sample = residuals[:5000] if len(residuals) > 5000 else residuals
    _, pvalue = shapiro(sample)
    passed = pvalue >= alpha
    return {
        "check"     : "Shapiro-Wilk (Normality)",
        "value"     : f"p = {pvalue:.4f}",
        "threshold" : f">= {alpha}",
        "passed"    : passed,
        "advice"    : "✅ Residuals are approximately normal." if passed
                      else "❌ Residuals not normal → try log/Box-Cox transform.",
    }


def check_stationarity(train_series, alpha=0.05):
    """ADF test: data should be stationary."""
    _, pvalue, *_ = adfuller(train_series)
    passed = pvalue <= alpha
    return {
        "check"     : "ADF Test (Stationarity)",
        "value"     : f"p = {pvalue:.4f}",
        "threshold" : f"<= {alpha}",
        "passed"    : passed,
        "advice"    : "✅ Data is stationary." if passed
                      else "❌ Non-stationary → apply differencing (d=1 or D=1).",
    }


def check_residual_skewness(residuals, threshold=1.0):
    """Skewness of residuals should be close to 0."""
    sk     = skew(residuals)
    passed = abs(sk) <= threshold
    return {
        "check"     : "Residual Skewness",
        "value"     : f"{sk:.4f}",
        "threshold" : f"|skew| <= {threshold}",
        "passed"    : passed,
        "advice"    : "✅ Residuals are not heavily skewed." if passed
                      else f"❌ Skewed ({sk:.2f}) → apply log1p or Box-Cox transform.",
    }


def check_mae(actual, forecast, threshold_pct=20.0):
    """MAE as % of non-zero actual mean."""
    mae      = mean_absolute_error(actual, forecast)
    base     = actual[actual > 0].mean()
    mae_pct  = (mae / base) * 100 if base != 0 else float("inf")
    passed   = mae_pct <= threshold_pct
    return {
        "check"     : "MAE % (vs non-zero mean)",
        "value"     : f"{mae:.4f}  ({mae_pct:.1f}%)",
        "threshold" : f"<= {threshold_pct}%",
        "passed"    : passed,
        "advice"    : "✅ MAE is acceptable." if passed
                      else f"❌ MAE too high ({mae_pct:.1f}%) → model misses average magnitude.",
    }


# ═══════════════════════════════════════════════════════════════
#  TWO-STAGE SPECIFIC CHECKS
# ═══════════════════════════════════════════════════════════════

def check_zero_classifier(actual, zero_pred, alpha=0.70):
    """Classifier accuracy on zero vs non-zero."""
    actual_bin = (actual > 0).astype(int)
    correct    = (actual_bin == zero_pred).mean()
    passed     = correct >= alpha
    return {
        "check"     : "Zero Classifier Accuracy",
        "value"     : f"{correct * 100:.2f}%",
        "threshold" : f">= {alpha * 100:.0f}%",
        "passed"    : passed,
        "advice"    : f"✅ Classifier accuracy is good ({correct*100:.1f}%)." if passed
                      else f"❌ Classifier accuracy low ({correct*100:.1f}%) → add more lag/streak features.",
    }


def check_zero_recall(actual, zero_pred, threshold=0.65):
    """
    Recall on non-zero class — are we catching actual non-zero periods?
    Missing a non-zero (predicting 0 when actual > 0) is costly.
    """
    actual_bin  = (actual > 0).astype(int)
    true_pos    = ((zero_pred == 1) & (actual_bin == 1)).sum()
    actual_pos  = actual_bin.sum()
    recall      = true_pos / actual_pos if actual_pos > 0 else 0
    passed      = recall >= threshold
    return {
        "check"     : "Non-Zero Recall (catching actual spikes)",
        "value"     : f"{recall * 100:.2f}%",
        "threshold" : f">= {threshold * 100:.0f}%",
        "passed"    : passed,
        "advice"    : f"✅ Non-zero recall is good ({recall*100:.1f}%)." if passed
                      else f"❌ Missing real non-zero periods ({recall*100:.1f}%) → lower clf threshold or add features.",
    }


def check_zero_precision(actual, zero_pred, threshold=0.65):
    """
    Precision on non-zero class — when we predict non-zero, are we right?
    False positives (predicting non-zero when actual=0) inflate RMSE.
    """
    actual_bin  = (actual > 0).astype(int)
    true_pos    = ((zero_pred == 1) & (actual_bin == 1)).sum()
    pred_pos    = zero_pred.sum()
    precision   = true_pos / pred_pos if pred_pos > 0 else 0
    passed      = precision >= threshold
    return {
        "check"     : "Non-Zero Precision (false alarm rate)",
        "value"     : f"{precision * 100:.2f}%",
        "threshold" : f">= {threshold * 100:.0f}%",
        "passed"    : passed,
        "advice"    : f"✅ Precision is good ({precision*100:.1f}%)." if passed
                      else f"❌ Too many false non-zero predictions ({precision*100:.1f}%) → raise clf threshold.",
    }


def check_spike_capture(actual, forecast, spike_pct=0.90, tolerance=0.50):
    """
    Check if the model captures the top 10% largest actual values within 50%.
    These are the most important points for discount modelling.
    """
    spike_threshold = np.quantile(actual[actual > 0], spike_pct)
    spike_mask      = actual >= spike_threshold
    if spike_mask.sum() == 0:
        return {"check": "Spike Capture", "value": "N/A", "threshold": "N/A",
                "passed": True, "advice": "✅ No spikes detected."}

    spike_actual   = actual[spike_mask]
    spike_forecast = forecast[spike_mask]
    rel_error      = np.abs(spike_actual - spike_forecast) / (spike_actual + 1e-9)
    avg_rel_error  = rel_error.mean()
    passed         = avg_rel_error <= tolerance

    return {
        "check"     : f"Spike Capture (top {int((1-spike_pct)*100)}% values)",
        "value"     : f"avg relative error = {avg_rel_error*100:.1f}%",
        "threshold" : f"<= {tolerance*100:.0f}%",
        "passed"    : passed,
        "advice"    : f"✅ Spikes captured reasonably well." if passed
                      else f"❌ Spikes under-predicted ({avg_rel_error*100:.1f}% error) → try spike boost or higher quantile.",
    }


def check_improvement_over_sarima(rmse_two_stage, rmse_sarima, min_improvement=0.20):
    """Two-Stage should beat SARIMA by at least min_improvement %."""
    improvement = (rmse_sarima - rmse_two_stage) / rmse_sarima
    passed      = improvement >= min_improvement
    return {
        "check"     : "Improvement over SARIMA alone",
        "value"     : f"{improvement * 100:.1f}% RMSE reduction",
        "threshold" : f">= {min_improvement * 100:.0f}% reduction",
        "passed"    : passed,
        "advice"    : f"✅ Two-Stage improves RMSE by {improvement*100:.1f}% over SARIMA." if passed
                      else f"❌ Two-Stage only improves by {improvement*100:.1f}% → classifier may be hurting, not helping.",
    }


# ═══════════════════════════════════════════════════════════════
#  SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════

def _print_summary(results, title, metrics):
    passed_count = sum(1 for r in results if r["passed"])
    total        = len(results)

    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print(f"  {'Score':<20}: {passed_count}/{total} checks passed")
    print("=" * 65)

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n  [{status}]  {r['check']}")
        print(f"           Value: {r['value']}  |  Threshold: {r['threshold']}")
        print(f"           {r['advice']}")

    print("\n" + "=" * 65)
    if passed_count == total:
        print("  🎉 Model looks GOOD.")
    elif passed_count >= total * 0.7:
        print("  ⚠️  Model is ACCEPTABLE — minor improvements possible.")
    else:
        print("  🚨 Model NEEDS WORK — see advice above.")
    print("=" * 65 + "\n")


# ═══════════════════════════════════════════════════════════════
#  PREPROCESSING RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════

def _recommend(results):
    failed = [r for r in results if not r["passed"]]
    if not failed:
        return
    print("📋 RECOMMENDED NEXT STEPS:")
    print("-" * 45)
    fixes = set()
    for r in failed:
        name = r["check"]
        if "RMSE"        in name: fixes.add("→ Apply log1p transform or add exogenous features")
        if "Bias"        in name: fixes.add("→ Check for missing seasonal terms; try D=1")
        if "Autocorr"    in name: fixes.add("→ Increase p/q range or set d=1")
        if "Normality"   in name: fixes.add("→ Apply log1p or Box-Cox transform")
        if "Stationary"  in name: fixes.add("→ Set d=1 in SARIMA order")
        if "Skewness"    in name: fixes.add("→ Apply log1p: np.log1p(train['y'])")
        if "Classifier"  in name: fixes.add("→ Add streak/rolling features to classifier")
        if "Recall"      in name: fixes.add("→ Lower classifier decision threshold to 0.3–0.4")
        if "Precision"   in name: fixes.add("→ Raise classifier decision threshold to 0.6–0.7")
        if "Spike"       in name: fixes.add("→ Add spike_boost layer or quantile regressor")
        if "Improvement" in name: fixes.add("→ Check if classifier is mis-classifying too many points")
        if "MAE"         in name: fixes.add("→ Switch SARIMA to XGBoost for the value stage")
    for fix in fixes:
        print(f"  {fix}")
    print()


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC PLOTS — SARIMA
# ═══════════════════════════════════════════════════════════════

def _plot_sarima(train, test, forecast, residuals, best_params, rmse):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"SARIMA{best_params} — Diagnostic Report  |  RMSE: {rmse:.4f}",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(len(train)), train["y"].values, label="Train",     color="#2196F3", linewidth=1)
    ax1.plot(range(len(train), len(train)+len(test)), test["y"].values,  label="Actual",    color="#4CAF50", linewidth=1.5)
    ax1.plot(range(len(train), len(train)+len(test)), forecast,          label="Predicted", color="#FF9800", linewidth=1.5, linestyle="--")
    ax1.set_title("Full Timeline: Train / Actual / Predicted")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(residuals, color="#F44336", linewidth=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Residuals Over Time (should scatter around 0)")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(residuals, bins=30, color="#FF9800", edgecolor="black", alpha=0.8)
    ax3.set_title("Residual Distribution (should be ~normal)")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    plot_acf(residuals, ax=ax4, lags=20, color="#1976D2")
    ax4.set_title("ACF of Residuals (should be ~0 after lag 0)")

    ax5 = fig.add_subplot(gs[2, 1])
    plot_pacf(residuals, ax=ax5, lags=20, color="#1976D2")
    ax5.set_title("PACF of Residuals (should be ~0 after lag 0)")

    plt.savefig("sarima_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Saved: sarima_diagnostics.png")


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC PLOTS — TWO-STAGE
# ═══════════════════════════════════════════════════════════════

def _plot_two_stage(train, test, actual, final_forecast, sarima_forecast,
                    zero_pred, zero_pred_proba, residuals, rmse, best_params):

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(f"Two-Stage Model — Diagnostic Report  |  RMSE: {rmse:.4f}",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. Full timeline ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(len(train)), train["y"].values,  label="Train",        color="#2196F3", linewidth=1)
    ax1.plot(range(len(train), len(train)+len(actual)), actual,            label="Actual",       color="#4CAF50", linewidth=1.5)
    ax1.plot(range(len(train), len(train)+len(final_forecast)), final_forecast, label="Two-Stage",    color="#9C27B0", linewidth=1.5, linestyle="-.")
    if sarima_forecast is not None:
        ax1.plot(range(len(train), len(train)+len(sarima_forecast)), sarima_forecast, label="SARIMA alone", color="#FF9800", linewidth=1, linestyle="--", alpha=0.7)
    ax1.set_title("Full Timeline: Train / Actual / Two-Stage / SARIMA")
    ax1.legend(); ax1.grid(alpha=0.3)

    # ── 2. Zero/Non-Zero classification ──
    ax2 = fig.add_subplot(gs[1, :])
    actual_bin = (actual > 0).astype(int)
    ax2.plot(actual_bin,       label="Actual Non-Zero",     color="#4CAF50", linewidth=1.2)
    ax2.plot(zero_pred,        label="Predicted Non-Zero",  color="#F44336", linewidth=1, linestyle="--")
    if zero_pred_proba is not None:
        ax2.plot(zero_pred_proba,  label="Proba Non-Zero",  color="#2196F3", alpha=0.5, linewidth=0.8)
    ax2.set_title("Stage 1 — Zero/Non-Zero Classification")
    ax2.legend(); ax2.grid(alpha=0.3)

    # ── 3. Residuals over time ──
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(residuals, color="#F44336", linewidth=0.8)
    ax3.axhline(0, color="black", linestyle="--", linewidth=1)
    ax3.set_title("Residuals Over Time (should scatter around 0)")
    ax3.grid(alpha=0.3)

    # ── 4. Residual histogram ──
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(residuals, bins=30, color="#FF9800", edgecolor="black", alpha=0.8)
    ax4.set_title("Residual Distribution (should be ~normal)")
    ax4.grid(alpha=0.3)

    # ── 5. Confusion matrix ──
    ax5 = fig.add_subplot(gs[3, 0])
    cm  = confusion_matrix(actual_bin, zero_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Zero", "Non-Zero"])
    disp.plot(ax=ax5, colorbar=False, cmap="Blues")
    ax5.set_title("Stage 1 — Confusion Matrix")

    # ── 6. Actual vs Predicted scatter ──
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.scatter(actual, final_forecast, alpha=0.3, color="#9C27B0", s=10)
    max_val = max(actual.max(), final_forecast.max())
    ax6.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1, label="Perfect fit")
    ax6.set_xlabel("Actual")
    ax6.set_ylabel("Predicted")
    ax6.set_title("Actual vs Predicted (closer to diagonal = better)")
    ax6.legend(); ax6.grid(alpha=0.3)

    plt.savefig("two_stage_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Saved: two_stage_diagnostics.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN — SARIMA CHECK
# ═══════════════════════════════════════════════════════════════

def check_model_performance(model_name, model_fit, train, test, forecast, best_params,
                             rmse_threshold=20.0, plot=True):
    """
    Full performance check for SARIMA model.

    Parameters
    ----------
    model_fit      : fitted SARIMAX model object
    train          : DataFrame with column 'y'
    test           : DataFrame with column 'y'
    forecast       : array-like, predicted values (same length as test)
    best_params    : tuple, e.g. (p, d, q, P, D, Q, m)
    rmse_threshold : RMSE% threshold (default 20)
    plot           : whether to show diagnostic plots
    """
    forecast  = np.array(forecast)
    residuals = test["y"].values - forecast
    rmse      = np.sqrt(np.mean(residuals ** 2))
    aic       = model_fit.aic

    results = [
        check_rmse_percent(rmse, test["y"].values, threshold=rmse_threshold),
        check_residual_mean(residuals),
        check_ljung_box(residuals),
        check_shapiro(residuals),
        check_stationarity(train["y"]),
        check_residual_skewness(residuals),
        check_mae(test["y"].values, forecast),
        check_spike_capture(test["y"].values, forecast),
    ]

    _print_summary(results, "SARIMA MODEL PERFORMANCE REPORT", {
        "Best Params" : str(best_params),
        "RMSE"        : f"{rmse:.4f}",
        "AIC"         : f"{aic:.4f}",
    })
    _recommend(results)

    if plot:
        _plot_sarima(train, test, forecast, residuals, best_params, rmse)

    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN — TWO-STAGE CHECK
# ═══════════════════════════════════════════════════════════════

def check_two_stage_performance(
    clf,
    regressor,
    train,
    test,
    final_forecast,
    zero_pred,
    zero_pred_proba,
    sarima_forecast=None,
    best_params=None,
    rmse_threshold=20.0,
    improvement_threshold=0.20,
    plot=True
):
    """
    Full performance check for Two-Stage model.

    Parameters
    ----------
    clf                  : fitted classifier (Stage 1)
    regressor            : fitted regressor/SARIMA (Stage 2)
    train                : DataFrame with column 'y'
    test                 : DataFrame with column 'y'
    final_forecast       : array-like, combined two-stage predictions
    zero_pred            : array-like, binary predictions from classifier (0 or 1)
    zero_pred_proba      : array-like, probability of non-zero from classifier
    sarima_forecast      : array-like, SARIMA-only forecast for comparison (optional)
    best_params          : tuple, SARIMA params used in Stage 2 (optional)
    rmse_threshold       : RMSE% threshold (default 20)
    improvement_threshold: minimum RMSE improvement over SARIMA (default 20%)
    plot                 : whether to show diagnostic plots
    """
    final_forecast   = np.array(final_forecast)
    zero_pred        = np.array(zero_pred)
    zero_pred_proba  = np.array(zero_pred_proba) if zero_pred_proba is not None else None

    # ── Align lengths ──
    min_len        = min(len(final_forecast), len(zero_pred), len(test["y"]))
    actual         = test["y"].values[-min_len:]
    final_forecast = final_forecast[-min_len:]
    zero_pred      = zero_pred[-min_len:]
    if zero_pred_proba is not None:
        zero_pred_proba = zero_pred_proba[-min_len:]
    if sarima_forecast is not None:
        sarima_forecast = np.array(sarima_forecast)[-min_len:]

    residuals = actual - final_forecast
    rmse      = np.sqrt(np.mean(residuals ** 2))
    mae       = mean_absolute_error(actual, final_forecast)

    # ── SARIMA baseline RMSE ──
    rmse_sarima = np.sqrt(mean_squared_error(actual, sarima_forecast)) \
                  if sarima_forecast is not None else None

    # ── Run all checks ──
    results = [
        # Regression quality
        check_rmse_percent(rmse, actual, threshold=rmse_threshold, nonzero_only=True),
        check_residual_mean(residuals),
        check_residual_skewness(residuals),
        check_ljung_box(residuals),
        check_shapiro(residuals),
        check_mae(actual, final_forecast),
        check_spike_capture(actual, final_forecast),
        # Classification quality
        check_zero_classifier(actual, zero_pred),
        
    ]

    # Add improvement check only if SARIMA baseline exists
    if rmse_sarima is not None:
        results.append(
            check_improvement_over_sarima(rmse, rmse_sarima, improvement_threshold)
        )

    # ── Metrics dict ──
    metrics = {
        "RMSE (Two-Stage)" : f"{rmse:.4f}",
        "MAE  (Two-Stage)" : f"{mae:.4f}",
        "Best Params"      : str(best_params) if best_params else "N/A",
    }
    if rmse_sarima is not None:
        metrics["RMSE (SARIMA)"]   = f"{rmse_sarima:.4f}"
        improvement = (rmse_sarima - rmse) / rmse_sarima * 100
        metrics["Improvement"]     = f"{improvement:.1f}%"
    mlflow.log_metric("Two-Stage RMSE", rmse)
    mlflow.log_metric("Two-Stage MAE", mae)
    mlflow.log_param("Two-Stage Best Params", str(best_params) if best_params else "N/A")
    
    _print_summary(results, "TWO-STAGE MODEL PERFORMANCE REPORT", metrics)

    # ── Classification report ──
    actual_bin = (actual > 0).astype(int)
    print("  Stage 1 — Classification Report:")
    print("  " + "-" * 45)
    report = classification_report(actual_bin, zero_pred,
                                   target_names=["Zero", "Non-Zero"])
    for line in report.split("\n"):
        print(f"  {line}")

    

    if plot:
        _plot_two_stage(
            train, test, actual, final_forecast, sarima_forecast,
            zero_pred, zero_pred_proba, residuals, rmse, best_params
        )

    return results


# ═══════════════════════════════════════════════════════════════
#  COMPARE BOTH MODELS SIDE BY SIDE
# ═══════════════════════════════════════════════════════════════

def compare_models(actual, sarima_forecast, two_stage_forecast, train=None, plot=True):
    """
    Quick side-by-side comparison of SARIMA vs Two-Stage.

    Parameters
    ----------
    actual              : array-like, ground truth
    sarima_forecast     : array-like, SARIMA predictions
    two_stage_forecast  : array-like, Two-Stage predictions
    train               : DataFrame with 'y' column (optional, for full timeline plot)
    plot                : whether to show comparison plot
    """
    actual             = np.array(actual)
    sarima_forecast    = np.array(sarima_forecast)
    two_stage_forecast = np.array(two_stage_forecast)

    min_len            = min(len(actual), len(sarima_forecast), len(two_stage_forecast))
    actual             = actual[-min_len:]
    sarima_forecast    = sarima_forecast[-min_len:]
    two_stage_forecast = two_stage_forecast[-min_len:]

    def metrics(pred):
        rmse    = np.sqrt(mean_squared_error(actual, pred))
        mae     = mean_absolute_error(actual, pred)
        nonzero = actual[actual > 0].mean()
        return rmse, mae, (rmse / nonzero * 100) if nonzero else float("inf")

    rmse_s, mae_s, pct_s = metrics(sarima_forecast)
    rmse_t, mae_t, pct_t = metrics(two_stage_forecast)

    print("\n" + "=" * 55)
    print("         MODEL COMPARISON")
    print("=" * 55)
    print(f"  {'Metric':<25} {'SARIMA':>12} {'Two-Stage':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'RMSE':<25} {rmse_s:>12.4f} {rmse_t:>12.4f}")
    print(f"  {'MAE':<25} {mae_s:>12.4f} {mae_t:>12.4f}")
    print(f"  {'RMSE % (non-zero mean)':<25} {pct_s:>11.1f}% {pct_t:>11.1f}%")
    improvement = (rmse_s - rmse_t) / rmse_s * 100
    winner      = "Two-Stage" if rmse_t < rmse_s else "SARIMA"
    print(f"\n  {'Winner':<25} {winner}")
    print(f"  {'RMSE Improvement':<25} {improvement:>11.1f}%")
    print("=" * 55 + "\n")

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        axes[0].plot(actual,             label="Actual",     color="green",  linewidth=1.5)
        axes[0].plot(sarima_forecast,    label=f"SARIMA (RMSE={rmse_s:.2f})",     color="orange", linewidth=1.2, linestyle="--")
        axes[0].plot(two_stage_forecast, label=f"Two-Stage (RMSE={rmse_t:.2f})",  color="blue",   linewidth=1.2, linestyle="-.")
        axes[0].set_title("SARIMA vs Two-Stage vs Actual")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        res_s = actual - sarima_forecast
        res_t = actual - two_stage_forecast
        axes[1].plot(res_s, label=f"SARIMA residuals    (MAE={mae_s:.2f})", color="orange", linewidth=0.8, alpha=0.8)
        axes[1].plot(res_t, label=f"Two-Stage residuals (MAE={mae_t:.2f})", color="blue",   linewidth=0.8, alpha=0.8)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Residuals Comparison")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  📊 Saved: model_comparison.png")

    return {"sarima": (rmse_s, mae_s), "two_stage": (rmse_t, mae_t), "improvement_pct": improvement}


# ═══════════════════════════════════════════════════════════════
#  EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════
"""
from sarima_diagnostics import (
    check_model_performance,
    check_two_stage_performance,
    compare_models
)

# ── SARIMA only ──
sarima_results = check_model_performance(
    model_fit   = best_model,
    train       = train,
    test        = test,
    forecast    = best_model.forecast(len(test)),
    best_params = best_params,
    plot        = True
)

# ── Two-Stage ──
two_stage_results = check_two_stage_performance(
    clf              = clf,
    regressor        = best_model,
    train            = train,
    test             = test,
    final_forecast   = final_forecast_hard,
    zero_pred        = zero_pred,
    zero_pred_proba  = zero_pred_proba,
    sarima_forecast  = sarima_aligned,
    best_params      = best_params,
    plot             = True
)

# ── Side-by-side comparison ──
compare_models(
    actual             = test["y"].values,
    sarima_forecast    = sarima_aligned,
    two_stage_forecast = final_forecast_hard,
    train              = train,
    plot               = True
)
"""