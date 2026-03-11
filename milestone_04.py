"""
Milestone 4: Predictive Modelling with Diagnostic Rigour
---------------------------------------------------------
This script fits a logistic regression model to predict premium conversion 
from user engagement metrics, while performing a rigorous four-step 
diagnostic check of the model assumptions.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import sys

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

DATA_USERS = 'data/finflow_users.csv'
DATA_TS = 'data/finflow_timeseries.csv'

try:
    df = pd.read_csv(DATA_USERS)
    ts_df = pd.read_csv(DATA_TS)
except FileNotFoundError as e:
    print(f"\n[ERROR] Missing Dataset: {e}")
    sys.exit(1)

# Ensure data is aligned (assuming row alignment as per requirement)
df_ordered = df.copy()
# In a real scenario, we'd merge on user_id, but here we follow requirements

# Add constant for intercept
X = sm.add_constant(df['score_views'])
y = df['premium_user']

# ============================================
# FIT LOGISTIC REGRESSION MODEL
# ============================================

model = sm.Logit(y, X).fit(disp=0)
summary = model.summary()

coef_intercept = model.params['const']
coef_score_views = model.params['score_views']

# ============================================
# REGRESSION ASSUMPTION DIAGNOSTICS
# ============================================

# 1. Independence (CRITICAL)
# Durbin-Watson requires residuals in time order
residuals = model.resid_pearson
dw_stat = durbin_watson(residuals)
independence_ok = 1.5 < dw_stat < 2.5

# 2. Linearity in Log-Odds (Box-Tidwell approximation)
# Interaction term: X * log(X). If p < 0.05, linearity is violated.
# Note: score_views might have zeros, so we use log(X + 1)
df_linearity = df.copy()
df_linearity['score_views_log'] = df_linearity['score_views'] * np.log(df_linearity['score_views'] + 1)
X_linear = sm.add_constant(df_linearity[['score_views', 'score_views_log']])
model_linear = sm.Logit(y, X_linear).fit(disp=0)
linearity_ok = model_linear.pvalues['score_views_log'] > 0.05

# 3. Homoscedasticity (Breusch-Pagan on deviance residuals)
# Breusch-Pagan is usually for OLS, but applied here to detect non-constant variance in residuals
deviance_resid = model.resid_dev
_, bp_p, _, _ = het_breuschpagan(deviance_resid, X)
homoscedasticity_ok = bp_p > 0.05

# 4. Normality (Shapiro-Wilk on Pearson residuals)
# Pearson residuals should be approximately normal for valid inference in larger samples
_, norm_p = stats.shapiro(residuals)
normality_ok = norm_p > 0.05

# ============================================
# GENERATE PREDICTIONS WITH UNCERTAINTY
# ============================================

score_views_new = 7
exog_new = [1.0, score_views_new]
prob_premium = model.predict(exog_new)[0]

# Delta Method for Prediction Interval (95%)
# PI = predict +/- 1.96 * sqrt(grad' * Cov * grad)
cov_matrix = model.cov_params()
grad = np.array([prob_premium * (1 - prob_premium), 
                score_views_new * prob_premium * (1 - prob_premium)])
se_delta = np.sqrt(grad.T @ cov_matrix @ grad)

margin_error = 1.96 * se_delta
pi_lower = max(0, prob_premium - margin_error)
pi_upper = min(1, prob_premium + margin_error)

# Determine minimum engagement threshold (50% tipping point)
# log(p/(1-p)) = 0 -> intercept + beta * X = 0 -> X = -intercept / beta
threshold_50 = -coef_intercept / coef_score_views if coef_score_views != 0 else np.nan

# ============================================
# VALIDATION CHECKS
# ============================================

assert 0 <= prob_premium <= 1, "Predicted probability must be between 0 and 1"
assert isinstance(linearity_ok, (bool, np.bool_)), "Linearity flag must be boolean"
assert isinstance(homoscedasticity_ok, (bool, np.bool_)), "Homoscedasticity flag must be boolean"
assert isinstance(normality_ok, (bool, np.bool_)), "Normality flag must be boolean"
assert isinstance(independence_ok, (bool, np.bool_)), "Independence flag must be boolean"

# ============================================
# RESULTS & INTERPRETATION
# ============================================

print("\n" + "="*80)
print(" STATISTICS MILESTONE 4: PREDICTIVE MODELLING REPORT ".center(80, "="))
print("="*80)

print(f"\nMODEL EQUATION:")
print(f"  log-odds(premium) = {coef_intercept:.3f} + {coef_score_views:.3f} * score_views")
print(f"  Odds Ratio (per view): {np.exp(coef_score_views):.3f}")

print(f"\nASSUMPTION DIAGNOSTICS:")
print("-" * 60)
print(f"{'Assumption':<20} {'Result':<12} {'Metric/P-value':<20}")
print("-" * 60)
print(f"{'Linearity (BT)':<20} {'OK' if linearity_ok else 'VIOLATED':<12} p = {model_linear.pvalues['score_views_log']:.4f}")
print(f"{'Homoscedasticity':<20} {'OK' if homoscedasticity_ok else 'VIOLATED':<12} p = {bp_p:.4f}")
print(f"{'Normality (SW)':<20} {'OK' if normality_ok else 'VIOLATED':<12} p = {norm_p:.4f}")
print(f"{'Independence (DW)':<20} {'OK' if independence_ok else 'VIOLATED':<12} DW = {dw_stat:.2f}")
print("-" * 60)

if not independence_ok:
    print(f"\n[WARNING] CRITICAL: Independence violation detected (DW={dw_stat:.2f}).")
    print("  Consequence: Standard errors are biased; p-values and CIs are UNRELIABLE.")
    print("  Remediation: Consider Clustered SEs or a Time-Series model (ARIMA/GARCH).")

print(f"\nPREDICTION (User with {score_views_new} views):")
print(f"  Predicted Probability: {prob_premium:.1%}")
print(f"  Approx. 95% PI:      ({pi_lower:.1%}, {pi_upper:.1%})")

print(f"\nBUSINESS RECOMMENDATION:")
print(f"  - Minimum engagement threshold for >50% conversion: {threshold_50:.1f} score views.")
print(f"  - Each additional score view multiplies conversion odds by {np.exp(coef_score_views):.2f}x.")

print(f"\nFINAL STRATEGIC VERDICT:")
if independence_ok and linearity_ok:
    print("  The model is statistically sound. Optimizing score views is a validated growth lever.")
else:
    print("  Model reliability is limited by diagnostic violations. Use as an directional indicator ")
    print("  rather than a precise forecasting tool until temporal/linear biases are addressed.")

print(f"\nSYNTHESIS:")
print("  - Connection to M1: Probabilities (P(Premium|Engagement)) are now parameterized rather than static.")
print("  - Connection to M2: Heavy-tailed engagement distributions (Kurtosis) explain probability clusters.")
print("="*80 + "\n")
