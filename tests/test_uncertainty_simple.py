#!/usr/bin/env python3
"""
Standalone test for uncertainty metrics calculation.
Tests the mathematical formulas directly.
"""


def calculate_uncertainty_metrics(yes_prob: float, no_prob: float):
    """
    Calculate uncertainty metrics for resampling decisions.

    Returns:
        coverage: Probability mass on yes/no tokens (0 to 1)
        binary_uncertainty: Uncertainty between yes and no (0 to 1, higher = more uncertain)
        eval_worthiness: Combined score indicating if resampling might help (0 to 1)
    """
    # Coverage: how much probability mass is on yes/no
    coverage = yes_prob + no_prob

    # Binary uncertainty: how uncertain between yes and no
    if coverage > 1e-8:
        yes_norm = yes_prob / coverage
        no_norm = no_prob / coverage
        binary_uncertainty = 1.0 - abs(yes_norm - no_norm)
    else:
        # Model doesn't engage with yes/no at all
        binary_uncertainty = 1.0  # Maximally uncertain

    # Eval worthiness: worth resampling when model engages AND is uncertain
    # High when: coverage is high (model thinks question is relevant)
    #            AND binary_uncertainty is high (model can't decide)
    eval_worthiness = coverage * binary_uncertainty

    return coverage, binary_uncertainty, eval_worthiness


def test_uncertainty_metrics():
    """Test various scenarios of uncertainty calculation."""

    # Test scenarios
    test_cases = [
        {
            "name": "Very certain YES (high coverage, low uncertainty)",
            "yes_prob": 0.90,
            "no_prob": 0.08,
        },
        {
            "name": "Very certain NO (high coverage, low uncertainty)",
            "yes_prob": 0.05,
            "no_prob": 0.92,
        },
        {
            "name": "Uncertain 50/50 (high coverage, high uncertainty)",
            "yes_prob": 0.48,
            "no_prob": 0.45,
        },
        {
            "name": "Doesn't engage (low coverage)",
            "yes_prob": 0.05,
            "no_prob": 0.03,
        },
        {
            "name": "Moderately uncertain (medium coverage, medium uncertainty)",
            "yes_prob": 0.65,
            "no_prob": 0.30,
        },
        {
            "name": "No engagement at all",
            "yes_prob": 0.01,
            "no_prob": 0.01,
        }
    ]

    print("=" * 80)
    print("Testing Uncertainty Metrics")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        yes_prob = test["yes_prob"]
        no_prob = test["no_prob"]
        other_prob = 1.0 - yes_prob - no_prob

        coverage, binary_uncertainty, eval_worthiness = calculate_uncertainty_metrics(
            yes_prob, no_prob
        )

        print(f"\n{i}. {test['name']}")
        print(f"   Input: yes={yes_prob:.2f}, no={no_prob:.2f}, other={other_prob:.2f}")
        print(f"   Results:")
        print(f"     - coverage:           {coverage:.4f}")
        print(f"     - binary_uncertainty: {binary_uncertainty:.4f}")
        print(f"     - eval_worthiness:    {eval_worthiness:.4f}")

        # Interpretation for resampling
        if eval_worthiness > 0.6:
            decision = "🔴 HIGH: Strongly consider resampling"
        elif eval_worthiness > 0.3:
            decision = "🟡 MEDIUM: Consider resampling if confidence_score is bad"
        else:
            decision = "🟢 LOW: Skip resampling (model is certain or doesn't engage)"

        print(f"   → {decision}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("  - eval_worthiness = coverage × binary_uncertainty")
    print("  - High eval_worthiness → Model engages with yes/no AND is uncertain")
    print("  - Low eval_worthiness → Either model is certain OR doesn't think in yes/no terms")
    print("=" * 80)


if __name__ == "__main__":
    test_uncertainty_metrics()
