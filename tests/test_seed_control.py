#!/usr/bin/env python3
"""
Test if seed control makes different strategies produce identical outputs.

This script runs a small experiment (3 samples, 2 strategies) to verify:
1. With seed control, do all strategies produce the same final answers?
2. If they still differ, there's a real bug in the generation code.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.llama_align.evaluation.experiment_runner import FrequencyAblationRunner

def main():
    print("="*80)
    print("SEED CONTROL TEST: Do strategies produce identical outputs with same seed?")
    print("="*80)

    # Load small model for fast testing
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Create experiment runner
    runner = FrequencyAblationRunner(
        model=model,
        tokenizer=tokenizer,
        output_dir="../../test_results/seed_control_test"
    )

    # Test with 2 strategies on 3 samples
    strategy_configs = [
        {"name": "every_n", "n": 3},
        {"name": "every_n", "n": 5},
    ]

    print("\nRunning experiment with seed control...")
    print("- Dataset: gsm8k")
    print("- Samples: 3")
    print("- Strategies: every_3, every_5")
    print("\nIf seed control works: All samples should have IDENTICAL answers across strategies")
    print("If bug exists: Some samples will still have DIFFERENT answers across strategies")
    print("="*80)

    results = runner.run_experiment(
        dataset_name="gsm8k",
        strategy_configs=strategy_configs,
        suffix_category="standard",
        suffix_index=0,
        max_samples=3,
        max_new_tokens=512,
    )

    # Analyze if strategies produced identical outputs
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)

    # Strategy names are generated from get_name(), e.g., "every_3_tokens", "every_5_tokens"
    strategy_names = list(results.keys())
    if "comparison" in strategy_names:
        strategy_names.remove("comparison")
    num_samples = len(results[strategy_names[0]]["traces"])

    identical_count = 0
    different_count = 0

    for sample_idx in range(num_samples):
        answers = []
        for strategy_name in strategy_names:
            trace = results[strategy_name]["traces"][sample_idx]
            answers.append(trace["final_answer"])

        # Check if all answers are identical
        if len(set(answers)) == 1:
            identical_count += 1
            status = "✓ IDENTICAL"
        else:
            different_count += 1
            status = "✗ DIFFERENT"

        print(f"\nSample {sample_idx}: {status}")
        for strategy_name, answer in zip(strategy_names, answers):
            # Truncate long answers for display
            display_answer = answer[:100] + "..." if len(answer) > 100 else answer
            print(f"  {strategy_name:20s}: {display_answer}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Identical outputs: {identical_count}/{num_samples}")
    print(f"Different outputs: {different_count}/{num_samples}")

    if different_count == 0:
        print("\n✓ SUCCESS: Seed control works! All strategies produce identical outputs.")
        print("  → The different outputs in your experiments were due to random variation.")
        print("  → With seed control, you can now fairly compare strategies.")
    else:
        print("\n✗ BUG DETECTED: Strategies still produce different outputs with same seed!")
        print("  → This confirms there's a real bug in the generation code.")
        print("  → Likely causes:")
        print("    1. Cache handling differs between evaluation frequencies")
        print("    2. Attention mask corruption")
        print("    3. Numerical drift in repeated cache operations")
        print("    4. Logits indexing issue (despite logits_to_keep)")

    print("="*80)

if __name__ == "__main__":
    main()
