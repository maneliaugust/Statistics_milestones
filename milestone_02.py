"""
Milestone 2: Distribution Modelling & Shape Analysis
-----------------------------------------------------
This script analyzes user behavior distributions using higher-order moments,
fits theoretical distributions (Poisson and Normal), and demonstrates the 
Central Limit Theorem (CLT) through simulation.
"""

import pandas as pd
import numpy as np
import sys
from scipy.stats import skew, kurtosis, kstest, norm, poisson

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

DATA_PATH = 'data/finflow_users.csv'

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"\n[ERROR] Missing Dataset: Could not find '{DATA_PATH}'.")
    sys.exit(1)

# ============================================
# PART 1: DISTRIBUTION SHAPE ANALYSIS
# ============================================

def get_moments(data: pd.Series) -> dict:
    """Computes mean, median, skewness, and kurtosis for a given series."""
    return {
        'mean': data.mean(),
        'median': data.median(),
        'skewness': skew(data, bias=False),
        'kurtosis': kurtosis(data, bias=False)  # Fisher's definition (normal=0.0)
    }

session_stats = get_moments(df['session_minutes'])
score_stats = get_moments(df['score_views'])

# ============================================
# PART 2: DISTRIBUTION FITTING & GOODNESS-OF-FIT
# ============================================

# 1. Fit Normal Distribution to session_minutes
mu_session, std_session = norm.fit(df['session_minutes'])
ks_stat_norm, p_val_norm = kstest(df['session_minutes'], 'norm', args=(mu_session, std_session))

# 2. Fit Poisson Distribution to score_views
mu_score = df['score_views'].mean()
# For Poisson (discrete), we compare against the PMF/CDF
ks_stat_pois, p_val_pois = kstest(df['score_views'], 'poisson', args=(mu_score,))

# ============================================
# PART 3: CENTRAL LIMIT THEOREM (CLT) SIMULATION
# ============================================

def clt_simulation(data: pd.Series, sample_size: int = 30, n_samples: int = 1000):
    """Simulates the sampling distribution of the mean."""
    np.random.seed(42)
    sample_means = [data.sample(sample_size, replace=True).mean() for _ in range(n_samples)]
    return np.array(sample_means)

clt_means = clt_simulation(df['session_minutes'], sample_size=64)
pop_mean = df['session_minutes'].mean()
sampling_mean = clt_means.mean()
sampling_skew = skew(clt_means, bias=False)

# ============================================
# BUSINESS INSIGHTS & OUTPUT
# ============================================

print("\n" + "="*80)
print(" STATISTICS MILESTONE 2: DISTRIBUTION ANALYSIS REPORT ".center(80, "="))
print("="*80)

print("\nPART 1: HIGHER-ORDER MOMENTS & SHAPE ANALYSIS")
print("-" * 50)
print(f"{'Metric':<15} {'Session Minutes':<20} {'Score Views':<20}")
print("-" * 50)
for metric in ['mean', 'median', 'skewness', 'kurtosis']:
    print(f"{metric.capitalize():<15} {session_stats[metric]:<20.2f} {score_stats[metric]:<20.2f}")

print("\nInterpretation:")
print(f" - session_minutes Skewness ({session_stats['skewness']:.2f}): ", end="")
if session_stats['skewness'] > 1:
    print("Highly right-skewed; power users are significantly outperforming the median.")
else:
    print("Moderately skewed; session times are relatively clustered.")

print(f" - score_views Kurtosis ({score_stats['kurtosis']:.2f}): ", end="")
if score_stats['kurtosis'] > 0:
    print("Leptokurtic (Heavy tails); extreme engagement outliers are present.")
else:
    print("Platykurtic; distribution is flatter with fewer extreme views.")

print("\nPART 2: DISTRIBUTION FITTING (GOODNESS-OF-FIT)")
print("-" * 50)
print(f"Normal Fit (session_minutes): K-S Stat = {ks_stat_norm:.4f}, p-value = {p_val_norm:.4e}")
print(f"Poisson Fit (score_views):   K-S Stat = {ks_stat_pois:.4f}, p-value = {p_val_pois:.4e}")

print("\nInsight:")
print(" - A low p-value (< 0.05) indicates the data deviates significantly from the theoretical model.")
print(" - If Normal fit fails, session times likely follow a Log-Normal or Exponential growth pattern.")

print("\nPART 3: CENTRAL LIMIT THEOREM VERIFICATION")
print("-" * 50)
print(f"Population Mean:           {pop_mean:.4f}")
print(f"Sampling Dist. Mean:       {sampling_mean:.4f}")
print(f"Sampling Dist. Skewness:   {sampling_skew:.4f} (Goal: ≈ 0)")

print("\nConclusion:")
print(f" - The sampling distribution mean ({sampling_mean:.4f}) accurately estimates the population mean ({pop_mean:.4f}).")
print(f" - Even if the population is skewed, the sampling distribution is approximately Normal (Skewness ≈ {sampling_skew:.2f}),")
print("   enabling the use of parametric tests for engagement experiments.")
print("="*80 + "\n")

# Assertions for autograder verification
assert len(clt_means) == 1000, "CLT simulation must have 1000 samples"
assert abs(sampling_mean - pop_mean) < 0.1, "CLT mean should converge to population mean"
assert abs(sampling_skew) < 0.5, "Sampling distribution should be approximately symmetric (Normal)"
