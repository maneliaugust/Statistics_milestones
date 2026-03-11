"""
Milestone 3: Statistical Inference & Hypothesis Testing
-------------------------------------------------------
This script performs inferential statistics on user behavior data, including
confidence intervals, bootstrap resampling, t-tests, chi-square tests, 
and A/B test analysis with Bonferroni correction.
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

DATA_USERS = 'data/finflow_users.csv'
DATA_AB = 'data/finflow_ab_test.csv'

try:
    df = pd.read_csv(DATA_USERS)
    ab_df = pd.read_csv(DATA_AB)
except FileNotFoundError as e:
    print(f"\n[ERROR] Missing Dataset: {e}")
    sys.exit(1)

# ============================================
# PART 1: CONFIDENCE INTERVALS
# ============================================

# 95% CI for mean session duration (t-distribution)
n = len(df)
mean_minutes = df['session_minutes'].mean()
sd_minutes = df['session_minutes'].std(ddof=1)
se_minutes = sd_minutes / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
ci_mean_lower = mean_minutes - t_crit * se_minutes
ci_mean_upper = mean_minutes + t_crit * se_minutes
margin_error_mean = t_crit * se_minutes

# 95% CI for premium conversion rate (Wilson score interval)
successes = df['premium_user'].sum()
n_total = len(df)
p_hat = successes / n_total
z_crit = stats.norm.ppf(0.975)  # 1.96 for 95% CI

# Wilson Score Interval Formula
denominator = 1 + (z_crit**2 / n_total)
adj_p = p_hat + (z_crit**2 / (2 * n_total))
sd_term = z_crit * np.sqrt((p_hat * (1 - p_hat) / n_total) + (z_crit**2 / (4 * n_total**2)))

ci_prop_lower = (adj_p - sd_term) / denominator
ci_prop_upper = (adj_p + sd_term) / denominator
margin_error_prop = (ci_prop_upper - ci_prop_lower) / 2

# ============================================
# PART 2: BOOTSTRAP METHODS
# ============================================

session_minutes = df['session_minutes'].values
n_boot = 10000
np.random.seed(42)

# Bootstrap resampling for median
bootstrap_medians = np.median(np.random.choice(session_minutes, size=(n_boot, len(session_minutes)), replace=True), axis=1)

# 95% percentile CI
ci_boot_lower = np.percentile(bootstrap_medians, 2.5)
ci_boot_upper = np.percentile(bootstrap_medians, 97.5)
point_estimate_median = np.median(session_minutes)

# ============================================
# PART 3: HYPOTHESIS TESTING I (T-TEST)
# ============================================

free_users = df[df['premium_user'] == 0]['session_minutes']
premium_users = df[df['premium_user'] == 1]['session_minutes']

h0_ttest = "H₀: μ_premium ≤ μ_free (premium users do not have longer sessions)"
ha_ttest = "Hₐ: μ_premium > μ_free (premium users have longer sessions)"

# Check Normality assumption (Shapiro-Wilk)
_, shapiro_free_p = stats.shapiro(free_users)
_, shapiro_premium_p = stats.shapiro(premium_users)
normality_ok = (shapiro_free_p > 0.05) and (shapiro_premium_p > 0.05)

# Check equal variance (Levene's test)
_, levene_p = stats.levene(free_users, premium_users)
equal_var = levene_p > 0.05

# Run Welch's t-test (one-tailed)
t_stat, p_two_tail = stats.ttest_ind(premium_users, free_users, equal_var=False)
p_value_ttest = p_two_tail / 2 if t_stat > 0 else 1 - p_two_tail / 2
reject_h0_ttest = p_value_ttest < 0.05

# Calculate Cohen's d (pooled SD for effect size)
n1, n2 = len(free_users), len(premium_users)
sd1, sd2 = free_users.std(ddof=1), premium_users.std(ddof=1)
pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1 + n2 - 2))
cohens_d = (premium_users.mean() - free_users.mean()) / pooled_sd

# Power analysis (Approximate n needed for 80% power)
# Rough formula for independent t-test with alpha=0.05: n ≈ 16 / d^2
n_needed_ttest = 16 / (cohens_d**2) if abs(cohens_d) > 0 else 0

# ============================================
# PART 4: HYPOTHESIS TESTING II (CHI-SQUARE)
# ============================================

contingency_table = pd.crosstab(df['risk_profile'], df['premium_user'])
chi2_stat, p_value_chi2, dof_chi2, expected = stats.chi2_contingency(contingency_table)

min_expected = expected.min()
assumption_met = min_expected >= 5 or (np.sum(expected >= 5) / expected.size >= 0.8 and min_expected >= 1)

n_chi2 = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2_stat / (n_chi2 * min(contingency_table.shape[0]-1, contingency_table.shape[1]-1)))

# ============================================
# PART 5: MULTIPLE COMPARISONS (A/B TEST)
# ============================================

conversion_rates = ab_df.groupby('variant')['converted'].mean()
control_rate = conversion_rates['control']

results = []
variants_to_test = ['variant_a', 'variant_b', 'variant_c', 'variant_d']
alpha = 0.05
m = len(variants_to_test)
alpha_adj = alpha / m

for variant in variants_to_test:
    v_data = ab_df[ab_df['variant'] == variant]
    c_data = ab_df[ab_df['variant'] == 'control']
    
    # Define counts for z-test
    successes_variant = v_data['converted'].sum()
    n_variant = len(v_data)
    successes_control = c_data['converted'].sum()
    n_control = len(c_data)
    
    # Run two-proportion z-test (manual implementation)
    # H0: p_variant <= p_control, Ha: p_variant > p_control
    p_v = successes_variant / n_variant
    p_c = successes_control / n_control
    p_pooled = (successes_variant + successes_control) / (n_variant + n_control)
    
    # Calculate z-statistic
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_variant + 1/n_control))
    z_stat = (p_v - p_c) / se_pooled if se_pooled > 0 else 0
    
    # One-tailed p-value (Ha: p_v > p_c)
    p_val = 1 - stats.norm.cdf(z_stat)
    
    significant = p_val < alpha_adj
    abs_lift = conversion_rates[variant] - control_rate
    rel_lift = (conversion_rates[variant] - control_rate) / control_rate
    
    results.append({
        'variant': variant,
        'conversion_rate': conversion_rates[variant],
        'p_value': p_val,
        'significant': significant,
        'abs_lift': abs_lift,
        'rel_lift': rel_lift
    })

results_df = pd.DataFrame(results)

# ============================================
# VALIDATION CHECKS
# ============================================

assert ci_mean_lower < mean_minutes < ci_mean_upper, "Mean CI must contain point estimate"
assert ci_prop_lower < p_hat < ci_prop_upper, "Proportion CI must contain point estimate"
assert ci_boot_lower < point_estimate_median < ci_boot_upper, "Bootstrap CI must contain median"
assert cohens_d > 0, "Cohen's d should be positive (premium > free)"
assert 0 <= p_value_chi2 <= 1, "p-value must be between 0 and 1"
assert len(results_df) == 4, "Must test 4 variants"

# ============================================
# RESULTS & INTERPRETATION
# ============================================

print("\n" + "="*80)
print(" STATISTICS MILESTONE 3: INFERENTIAL ANALYSIS REPORT ".center(80, "="))
print("="*80)

print("\nPART 1: CONFIDENCE INTERVALS (95%)")
print("-" * 60)
print(f"{'Metric':<25} {'Point Est.':<12} {'95% CI':<24} {'MoE'}")
print("-" * 60)
print(f"{'Mean Session (min)':<25} {mean_minutes:<12.1f} ({ci_mean_lower:>4.1f}, {ci_mean_upper:>4.1f}) {'±' + str(round(margin_error_mean,1)):>10}")
print(f"{'Premium Rate':<25} {p_hat:<12.1%} ({ci_prop_lower:>4.1%}, {ci_prop_upper:>4.1%}) {'±' + str(round(margin_error_prop*100,1)) + '%':>10}")
print(f"{'Median Session (boot)':<25} {point_estimate_median:<12.1f} ({ci_boot_lower:>4.1f}, {ci_boot_upper:>4.1f}) {'N/A':>10}")

print("\nINTERPRETATION:")
print(f" - Mean Session: We are 95% confident the true population mean is between {ci_mean_lower:.1f} and {ci_mean_upper:.1f} mins.")
print(f" - Premium Rate: In the worst-case scenario, the conversion rate is likely at least {ci_prop_lower:.1%}.")

print("\nPART 2: BOOTSTRAP VS PARAMETRIC COMPARISON")
print("-" * 60)
boot_width = ci_boot_upper - ci_boot_lower
param_width = ci_mean_upper - ci_mean_lower
print(f"{'Width Comparison':<25} {'Mean (Param)':<15} {'Median (Boot)':<15}")
print(f"{'CI Width':<25} {param_width:<15.2f} {boot_width:<15.2f}")
print(f"\nRecommendation: The median CI is {'narrower' if boot_width < param_width else 'wider'} than the mean CI, ")
print("providing a more robust estimate of the 'typical' user experience given the distribution skew.")

print("\nPART 3: HYPOTHESIS TEST (PREMIUM VS FREE)")
print("-" * 60)
print(f"Normality (Shapiro): Free p={shapiro_free_p:.3f}, Premium p={shapiro_premium_p:.3f} -> {'OK' if normality_ok else 'VIOLATED'}")
print(f"Variance (Levene):  p={levene_p:.3f} -> {'OK' if equal_var else 'VIOLATED (Used Welch)'}")
print(f"Welch's t-test:     t={t_stat:.2f}, p={p_value_ttest:.4f}")
print(f"Decision:           {'REJECT H0' if reject_h0_ttest else 'FAIL TO REJECT H0'}")
print(f"Effect Size (d):    {cohens_d:.2f} ({'small' if cohens_d < 0.2 else 'medium' if cohens_d < 0.5 else 'large'})")
print(f"Power Analysis:      n={n_needed_ttest:.0f} per group needed for 80% power.")

print("\nPART 4: CHI-SQUARE TEST (RISK VS PREMIUM)")
print("-" * 60)
print(f"χ² Stat: {chi2_stat:.2f}, p={p_value_chi2:.4f}, df={dof_chi2}")
print(f"Assumption Check: Min Expected Count = {min_expected:.1f} -> {'OK' if assumption_met else 'VIOLATED'}")
print(f"Effect Size (V):  {cramers_v:.3f} ({'weak' if cramers_v < 0.1 else 'moderate' if cramers_v < 0.3 else 'strong'})")

print("\nPART 5: A/B TEST (BONFERRONI CORRECTION)")
print("-" * 80)
print(f"Control Rate: {control_rate:.2%}")
print(f"Adjusted alpha: {alpha_adj:.4f}\n")
print(results_df.to_string(index=False, formatters={
    'conversion_rate': '{:.2%}'.format, 'p_value': '{:.4f}'.format,
    'abs_lift': '{:+.2%}'.format, 'rel_lift': '{:+.1%}'.format
}))

significant_variants = results_df[results_df['significant']]
if len(significant_variants) > 0:
    best = significant_variants.loc[significant_variants['abs_lift'].idxmax()]
    print(f"\nDEPLOY VARIANT {best['variant'].upper()}: Absolute lift of {best['abs_lift']:+.2%}. ")
    print("This result is statistically significant after correcting for multiple comparisons.")
else:
    print("\nNO VARIANTS SURVIVE CORRECTION. Iterate on design or collect more data.")
print("="*80 + "\n")
