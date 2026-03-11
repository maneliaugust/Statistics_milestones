"""
Milestone 2: Distribution Modelling & Shape Analysis
-----------------------------------------------------
This script analyzes user behavior distributions using higher-order moments,
fits theoretical distributions (Poisson and Normal), and demonstrates the 
Central Limit Theorem (CLT) through simulation.
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

DATA_PATH = 'data/finflow_users.csv'

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"\n[ERROR] Missing Dataset: Could not find '{DATA_PATH}'.")
    print("Please ensure the CSV file is located in the 'data/' subdirectory relative to this script.")
    sys.exit(1)

session_minutes = df['session_minutes'].values
score_views = df['score_views'].values

# ============================================
# PART 1: MOMENTS & SHAPE ANALYSIS
# ============================================

# Calculate mean
mean_minutes = np.mean(session_minutes)

# Calculate variance (use ddof=1 for sample variance)
variance_minutes = np.var(session_minutes, ddof=1)

# Calculate skewness (use bias=False for unbiased estimator)
skewness_minutes = stats.skew(session_minutes, bias=False)

# Calculate excess kurtosis (use bias=False; scipy returns excess by default)
kurtosis_minutes = stats.kurtosis(session_minutes, bias=False)

# ============================================
# PART 2: STANDARD DISTRIBUTIONS
# ============================================

# Fit Poisson distribution to score_views
# Lambda MLE for Poisson is simply the sample mean
lambda_poisson = np.mean(score_views)

# Fit Normal distribution to session_minutes
mu_normal, sigma_normal = stats.norm.fit(session_minutes)

# Perform KS test for Poisson fit
# stats.kstest requires a CDF function or a frozen distribution
ks_stat_poisson, p_value_poisson = stats.kstest(score_views, 'poisson', args=(lambda_poisson,))

# Perform KS test for Normal fit
ks_stat_normal, p_value_normal = stats.kstest(session_minutes, 'norm', args=(mu_normal, sigma_normal))

# ============================================
# PART 3: SAMPLING DISTRIBUTIONS & CLT
# ============================================

pop_mean = session_minutes.mean()
pop_std = session_minutes.std(ddof=1)
sample_sizes = [10, 30, 100]
n_reps = 10000
np.random.seed(42)

# Simulate sampling distributions
sampling_distributions = {}
for n in sample_sizes:
    # draw n_reps samples of size n, compute means
    sample_means = np.array([np.random.choice(session_minutes, size=n, replace=True).mean() for _ in range(n_reps)])
    sampling_distributions[n] = sample_means

# Calculate empirical SE for each n
empirical_ses = {}
for n, means in sampling_distributions.items():
    empirical_ses[n] = np.std(means, ddof=1)

# Calculate theoretical SE (sigma/sqrt(n))
theoretical_ses = {n: pop_std / np.sqrt(n) for n in sample_sizes}

# Determine minimum n for approximate Normality (|skew| < 0.5)
min_n_normal = None
for n in sample_sizes:
    s = stats.skew(sampling_distributions[n], bias=False)
    if abs(s) < 0.5:
        min_n_normal = n
        break

# ============================================
# VALIDATION CHECKS
# ============================================

assert mean_minutes > 0, "Mean must be positive"
assert variance_minutes > 0, "Variance must be positive"
assert lambda_poisson > 0, "Poisson lambda must be positive"
assert sigma_normal > 0, "Normal sigma must be positive"

for n in sample_sizes:
    error = abs(empirical_ses[n] - theoretical_ses[n]) / theoretical_ses[n]
    assert error < 0.1, f"SE mismatch for n={n}: empirical={empirical_ses[n]:.2f}, theoretical={theoretical_ses[n]:.2f}"

# ============================================
# RESULTS & INTERPRETATION
# ============================================

print("\n" + "="*60)
print(" SESSION DURATION MOMENTS ".center(60, "="))
print("="*60)
print(f"Mean:     {mean_minutes:.2f} minutes")
print(f"Variance: {variance_minutes:.2f} (SD = {variance_minutes**0.5:.2f})")
print(f"Skewness: {skewness_minutes:.2f}")
print(f"Kurtosis: {kurtosis_minutes:.2f} (excess)")

print("\nSHAPE INTERPRETATION:")
skew_desc = "highly right-skewed" if skewness_minutes > 1 else "moderately right-skewed" if skewness_minutes > 0.5 else "relatively symmetric"
print(f"  Skewness ({skewness_minutes:.2f}): Data is {skew_desc}. Most users have short sessions, ")
print("  while a few 'power users' stay active for exceptionally long durations.")

kurt_desc = "heavy-tailed (leptokurtic)" if kurtosis_minutes > 1 else "light-tailed" if kurtosis_minutes < -1 else "near-normal"
print(f"  Kurtosis ({kurtosis_minutes:.2f}): The distribution is {kurt_desc}, suggesting that ")
print("  extreme session lengths (outliers) are more frequent than in a normal distribution.")

print(f"\nBUSINESS IMPLICATION:")
print("  Product features should be optimized for the median session length to serve the majority, ")
print("  while specialized engagement tools can be targeted towards the high-value 'power user' tail.")

print("\n" + "="*60)
print(" DISTRIBUTION FITTING RESULTS ".center(60, "="))
print("="*60)
print(f"{'Distribution':<15} {'Parameter(s)':<25} {'KS Stat':<10} {'p-value':<10}")
print("-" * 60)
print(f"Poisson         λ = {lambda_poisson:.2f}{'':<15} {ks_stat_poisson:.3f}    {p_value_poisson:.3f}")
print(f"Normal          μ = {mu_normal:.2f}, σ = {sigma_normal:.2f}   {ks_stat_normal:.3f}    {p_value_normal:.3f}")
print("="*60)

print("\nGOODNESS-OF-FIT INTERPRETATION:")
p_pois_desc = "poor" if p_value_poisson < 0.05 else "adequate"
p_norm_desc = "poor" if p_value_normal < 0.05 else "adequate"
print(f"  Poisson fit: {p_pois_desc.capitalize()} fit (p={p_value_poisson:.3f}). Score views likely have more variance than mean.")
print(f"  Normal fit:  {p_norm_desc.capitalize()} fit (p={p_value_normal:.3f}). Session minutes are too skewed for a normal model.")

print(f"\nRECOMMENDATION FOR SIMULATION MODELS:")
print("  Use a Log-Normal or Gamma distribution for session durations to better capture the right-skew and ")
print("  zero-bound nature of the timing data in future discrete-event simulations.")

print("\n" + "="*60)
print(" CLT CONVERGENCE RESULTS ".center(60, "="))
print("="*60)
print(f"{'Sample Size (n)':<20} {'Empirical SE':<18} {'Theoretical SE':<18} {'Ratio'}")
print("-" * 60)
for n in sample_sizes:
    ratio = empirical_ses[n] / theoretical_ses[n]
    print(f"{n:<20} {empirical_ses[n]:<18.2f} {theoretical_ses[n]:<18.2f} {ratio:.3f}")
print("="*60)
print(f"\nMinimum n for approximate Normality: {min_n_normal}")

print(f"\nBUSINESS IMPLICATION:")
print(f"  For reliable A/B test results, a minimum sample size of n={min_n_normal} per variant is recommended. ")
print("  This ensures the sampling distribution of the mean is normal enough for standard Z-tests or t-tests.")
print("="*60 + "\n")
