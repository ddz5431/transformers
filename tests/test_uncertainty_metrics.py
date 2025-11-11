#!/usr/bin/env python3
"""
Test script for uncertainty metrics in LogitAnalyzer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformers/src'))

from transformers.generation.self_eval_logit_processor import LogitAnalyzer


def test_uncertainty_metrics():
    """Test various scenarios of uncertainty calculation."""

    # Create a mock analyzer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Mock: return single token IDs
            mock_ids = {"Yes": [1], " Yes": [2], "No": [3], " No": [4]}
            return mock_ids.get(text, [0])

        def decode(self, ids):
            return "mock"

    analyzer = LogitAnalyzer(
        tokenizer=MockTokenizer(),
        yes_tokens=["Yes", " Yes"],
        no_tokens=["No", " No"],
        device='cpu'
    )

    # Test scenarios
    test_cases = [
        {
            "name": "Very certain YES (high coverage, low uncertainty)",
            "yes_prob": 0.90,
            "no_prob": 0.08,
            "expected": {
                "coverage": 0.98,
                "binary_uncertainty": "~0.16",
                "eval_worthiness": "~0.16"
            }
        },
        {
            "name": "Very certain NO (high coverage, low uncertainty)",
            "yes_prob": 0.05,
            "no_prob": 0.92,
            "expected": {
                "coverage": 0.97,
                "binary_uncertainty": "~0.18",
                "eval_worthiness": "~0.17"
            }
        },
        {
            "name": "Uncertain 50/50 (high coverage, high uncertainty)",
            "yes_prob": 0.48,
            "no_prob": 0.45,
            "expected": {
                "coverage": 0.93,
                "binary_uncertainty": "~0.94",
                "eval_worthiness": "~0.87"
            }
        },
        {
            "name": "Doesn't engage (low coverage)",
            "yes_prob": 0.05,
            "no_prob": 0.03,
            "expected": {
                "coverage": 0.08,
                "binary_uncertainty": "~0.50",
                "eval_worthiness": "~0.04"
            }
        },
        {
            "name": "Moderately uncertain (medium coverage, medium uncertainty)",
            "yes_prob": 0.65,
            "no_prob": 0.30,
            "expected": {
                "coverage": 0.95,
                "binary_uncertainty": "~0.63",
                "eval_worthiness": "~0.60"
            }
        }
    ]

    print("=" * 80)
    print("Testing Uncertainty Metrics")
    print("=" * 80)

    for i, test in enumerate(test_cases, 1):
        yes_prob = test["yes_prob"]
        no_prob = test["no_prob"]

        coverage, binary_uncertainty, eval_worthiness = _calculate_uncertainty_for_eval(
            yes_prob, no_prob
        )

        print(f"\n{i}. {test['name']}")
        print(f"   Input: yes_prob={yes_prob:.2f}, no_prob={no_prob:.2f}")
        print(f"   Results:")
        print(f"     - coverage:           {coverage:.4f}")
        print(f"     - binary_uncertainty: {binary_uncertainty:.4f}")
        print(f"     - eval_worthiness:    {eval_worthiness:.4f}")
        print(f"   Expected: {test['expected']}")

        # Interpretation
        if eval_worthiness > 0.6:
            print(f"   → HIGH eval_worthiness: Strongly consider resampling")
        elif eval_worthiness > 0.3:
            print(f"   → MEDIUM eval_worthiness: Consider resampling based on confidence_score")
        else:
            print(f"   → LOW eval_worthiness: Skip resampling, model is certain or doesn't engage")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_uncertainty_metrics()
