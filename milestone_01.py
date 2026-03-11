"""
Milestone 1: Probability Foundations & Variable Types
-----------------------------------------------------
This script establishes the foundational probability metrics for the FinFlow user base.
It computes basic and conditional probabilities, verifies Bayes' Theorem, and 
classifies key random variables to inform product and engineering decisions.

Requirements handled:
- Robust FileNotFoundError handling.
- Basic, joint, and conditional probability calculations.
- Bayes' Theorem verification within tolerance.
- Odds ratio calculation and strategic recommendation.
- Comprehensive random variable classification.
"""

import pandas as pd
import sys
from typing import Dict, Any

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

DATA_PATH = 'data/finflow_users.csv'

try:
    # Load dataset with pandas
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"\n[ERROR] Missing Dataset: Could not find '{DATA_PATH}'.")
    print("Please ensure the CSV file is located in the 'data/' subdirectory relative to this script.")
    print("The autograder requires this specific structure to execute successfully.")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred while loading the data: {e}")
    sys.exit(1)

# ============================================
# PART 1: SAMPLE SPACES & BASIC PROBABILITY
# ============================================

# Define sample space for premium_user (Binary outcome)
sample_space_premium = set(df['premium_user'].unique())

# P(premium_user = 1): Baseline conversion rate
p_premium = (df['premium_user'] == 1).mean()

# P(score_views >= 5): High engagement threshold
p_high_engagement = (df['score_views'] >= 5).mean()

# P(risk_profile = 'aggressive'): Target segment for high-yield products
p_aggressive = (df['risk_profile'] == 'aggressive').mean()

# Joint probability P(score_views >= 5 AND premium_user = 1)
# Measures the overlap between high engagement and actual conversion
p_joint = ((df['score_views'] >= 5) & (df['premium_user'] == 1)).mean()

# ============================================
# PART 2: CONDITIONAL PROBABILITY & BAYES
# ============================================

# Define engagement masks for clarity
engaged_threshold = 3
engaged_mask = df['score_views'] >= engaged_threshold
premium_mask = df['premium_user'] == 1

# P(premium = 1 | score_views >= 3): Conversion rate among engaged users
p_premium_given_engaged = df.loc[engaged_mask, 'premium_user'].mean()

# P(score_views >= 3 | premium = 1): Engagement rate among premium users
p_engaged_given_premium = (df.loc[premium_mask, 'score_views'] >= engaged_threshold).mean()

# P(score_views >= 3): General engagement rate
p_engaged = engaged_mask.mean()

# Verify Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
# bayes_check should equal p_premium_given_engaged within 0.01 tolerance
bayes_check = (p_engaged_given_premium * p_premium) / p_engaged

# Calculate Odds Ratio
# Measures the strength of association between engagement and conversion
odds_engaged = p_premium_given_engaged / (1 - p_premium_given_engaged)
odds_base = p_premium / (1 - p_premium)
odds_ratio = odds_engaged / odds_base

# ============================================
# PART 3: RANDOM VARIABLE CLASSIFICATION
# ============================================

classifications: Dict[str, Dict[str, str]] = {
    'days_active': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, ..., 365}',
        'distribution': 'Poisson',
        'justification': 'Counts distinct days of activity; suitable for bounded counts with potential overdispersion.'
    },
    'score_views': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, 2, ...}',
        'distribution': 'Poisson',
        'justification': 'Count of independent score-viewing events occurring over a fixed observation window.'
    },
    'session_minutes': {
        'type': 'continuous',
        'support': 'non-negative real numbers [0, ∞)',
        'distribution': 'Log-Normal',
        'justification': 'Continuous time measurement; typically right-skewed as most sessions are short with a few very long outliers.'
    },
    'risk_profile': {
        'type': 'categorical',
        'support': "{'conservative', 'moderate', 'aggressive'}",
        'distribution': 'Categorical',
        'justification': 'Qualitative categories with no inherent mathematical order or distance between them.'
    },
    'premium_user': {
        'type': 'binary',
        'support': '{0, 1}',
        'distribution': 'Bernoulli',
        'justification': 'Models a single trial with two mutually exclusive outcomes (Success/Failure).'
    }
}

# ============================================
# VALIDATION ASSERTIONS
# ============================================

assert 0 <= p_premium <= 1, "P(premium) must be between 0 and 1"
assert 0 <= p_high_engagement <= 1, "P(high engagement) must be between 0 and 1"
assert 0 <= p_aggressive <= 1, "P(aggressive) must be between 0 and 1"
assert 0 <= p_joint <= 1, "Joint probability must be between 0 and 1"
assert abs(p_premium_given_engaged - bayes_check) < 0.01, "Bayes' theorem verification failed"
assert odds_ratio > 0, "Odds ratio must be positive"
assert all(classifications[var]['type'] for var in classifications), "All variables must be classified"

# ============================================
# BUSINESS INTERPRETATION & OUTPUT
# ============================================

print("\n" + "="*80)
print(" STATISTICS MILESTONE 1: BUSINESS INSIGHTS REPORT ".center(80, "="))
print("="*80)

print("\nPART 1: KEY PERFORMANCE INDICATORS")
print("-" * 40)
print(f"P(premium user):            {p_premium:.1%}")
print(f"  → Interpretation: Baseline conversion rate; {p_premium:.1%} of our current user base has opted for premium services.")
print(f"P(high engagement):         {p_high_engagement:.1%}")
print(f"  → Interpretation: {p_high_engagement:.1%} of users are highly active (5+ views), representing our core power users.")
print(f"P(aggressive risk profile): {p_aggressive:.1%}")
print(f"  → Interpretation: {p_aggressive:.1%} of users seek high risk/reward, a key target for sophisticated investment tools.")
print(f"Joint (Engaged & Premium):  {p_joint:.1%}")
print(f"  → Interpretation: Only {p_joint:.1%} of users are both high-engagement and premium, suggesting room for conversion growth.")

print("\nPART 2: BEHAVIORAL ANALYTICS & PREDICTION")
print("-" * 40)
print(f"P(premium | engaged):       {p_premium_given_engaged:.1%}")
print(f"P(engaged | premium):       {p_engaged_given_premium:.1%}")
print(f"Bayes Verification:         {bayes_check:.4f} ≈ {p_premium_given_engaged:.4f} [SUCCESS]")
print(f"\nEngagement Odds Ratio:      {odds_ratio:.2f}x")
print(f"  → Recommendation: Users with 3+ score views are {odds_ratio:.2f}x more likely to convert. ")
print(f"    Prioritize UI nudges that drive users to cross this specific viewing threshold to optimize acquisition costs.")

print("\nPART 3: RANDOM VARIABLE CLASSIFICATION")
print("-" * 80)
header = f"{'Variable':<20} {'Type':<15} {'Distribution':<20} {'Support'}"
print(header)
print("-" * 80)
for var, props in classifications.items():
    print(f"{var:<20} {props['type']:<15} {props['distribution']:<20} {props['support']}")

print("\n" + "="*80)
print("JUSTIFICATIONS")
print("-" * 20)
for var, props in classifications.items():
    print(f" - {var:<15}: {props['justification']}")

print("\nCRITICAL THINKING")
print("-" * 20)
print("Question: Why is score_views discrete but session_minutes continuous?")
print("Answer:   score_views represents a count of distinct events that occur in whole increments")
print("          (integers). In contrast, session_minutes measures time, which can be subdivided")
print("          infinitely into fractional components (real numbers) between any two points.")
print("="*80 + "\n")