#!/usr/bin/env python
"""Pipeline for single-objective molecular optimization using Qwen-2.5-7B-Instruct.

This script implements the evolutionary search algorithm described in
"Efficient Evolutionary Search Over Chemical Space with Large Language Models".
It integrates the helper functions from ``initialize_pool.py`` and
``run_iteration.py`` and optionally performs DPO fine tuning using the utilities
in ``rl_train.py``.

The reinforcement learning reward for each iteration combines three metrics:
1. ``new_reward_max - positive_reward_max``
2. ``new_reward_mean - positive_reward_mean``
3. ``diversity(new_samples)``
These metrics are weighted and summed to produce the final reward used for
optional training with Direct Preference Optimization.
"""

import argparse
import os
import pickle
from datetime import datetime

from initialize_pool import initialize_candidate_pool
from run_iteration import (
    calculate_rewards,
    sample_from_pool,
    prompt_llm,
    calculate_diversity,
    calculate_rl_reward,
)

# Optional imports used only when DPO training is requested
try:
    from rl_train import sample_trajectories, create_preference_pairs, train_with_dpo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:
    sample_trajectories = create_preference_pairs = train_with_dpo = None


def run_search(
    n,
    iterations,
    m,
    reward,
    vllm_endpoint,
    output_dir,
    sample_method="exp",
    model_path=None,
    hf_token=None,
    dpo=False,
    seed=42,
):
    """Run the single-objective evolutionary search.

    Parameters
    ----------
    n : int
        Size of the initial candidate pool.
    iterations : int
        Number of optimization iterations.
    m : int
        Number of positive/negative samples per iteration.
    reward : str
        Name of the reward oracle from TDC.
    vllm_endpoint : str
        Endpoint of the vLLM server hosting Qwen-2.5-7B-Instruct.
    output_dir : str
        Directory to store intermediate pools and, if DPO is enabled, the
        fine-tuned model.
    sample_method : str
        Sampling method for selecting positive/negative examples
        ("exp" or "uniform").
    model_path : str or None
        Path or HF identifier of the base model used for RL fine tuning.
    hf_token : str or None
        HuggingFace token for gated models.
    dpo : bool
        Whether to collect preference pairs and perform DPO training.
    seed : int
        Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ----- Step 1: initialise candidate pool -----
    candidate_pool = initialize_candidate_pool(n, seed)
    pool_file = os.path.join(output_dir, "pool_iteration_0.pkl")
    with open(pool_file, "wb") as f:
        pickle.dump(candidate_pool, f)

    # Containers for preference pairs if DPO is used
    all_preferred = []
    all_rejected = []

    for iteration in range(1, iterations + 1):
        print(f"\nIteration {iteration}/{iterations}")
        rewards = calculate_rewards(candidate_pool, reward)
        pos_samples, neg_samples = sample_from_pool(candidate_pool, rewards, m, sample_method)

        # Generate new molecules
        new_samples = prompt_llm(pos_samples, neg_samples, vllm_endpoint, m)
        new_rewards = calculate_rewards(new_samples, reward)
        diversity = calculate_diversity(new_samples)
        rl_reward = calculate_rl_reward(new_rewards, pos_samples, diversity)
        print(f"RL reward: {rl_reward:.4f}")

        # Update pool with unique molecules
        for smi in new_samples:
            if smi not in candidate_pool:
                candidate_pool.append(smi)

        # Save pool for this iteration
        pool_file = os.path.join(output_dir, f"pool_iteration_{iteration}.pkl")
        with open(pool_file, "wb") as f:
            pickle.dump(candidate_pool, f)

        if dpo and sample_trajectories is not None:
            # Recreate prompt used for generation to sample trajectories
            pos_fmt = "\n".join([f"SMILES: {s}, Reward: {r:.4f}" for s, r in pos_samples])
            neg_fmt = "\n".join([f"SMILES: {s}, Reward: {r:.4f}" for s, r in neg_samples])
            content = (
                f"Here are {m} positive samples with high rewards:\n{pos_fmt}\n\n"
                f"Here are {m} negative samples with low rewards:\n{neg_fmt}\n\n"
                f"Please analyze the results, and output {m} new SMILES strings for molecules "
                f"with rewards better than the positive samples. The new samples should be diversified."
            )
            prompt = (
                "<|im_start|>system\nYou are a black-box reward optimizer for molecular generation."
                "<|im_end|>\n<|im_start|>user\n" + content + "\n<|im_end|>\n<|im_start|>assistant"
            )
            trajectories = sample_trajectories(None, None, prompt, vllm_endpoint, num_samples=m)
            preferred, rejected = create_preference_pairs(trajectories, reward)
            all_preferred.extend(preferred)
            all_rejected.extend(rejected)

    # ----- Optional DPO training -----
    if dpo and all_preferred and model_path is not None and train_with_dpo is not None:
        print("\nStarting DPO fine tuning...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        train_with_dpo(model, tokenizer, all_preferred, all_rejected, os.path.join(output_dir, "dpo_model"))

    print("\nSearch completed. Results stored in", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Single-objective molecular optimization pipeline")
    parser.add_argument("--n", type=int, default=1000, help="Initial pool size")
    parser.add_argument("--iterations", type=int, default=3, help="Number of optimization iterations")
    parser.add_argument("--m", type=int, default=10, help="Number of positive/negative samples per iteration")
    parser.add_argument("--reward", type=str, default="qed", help="Reward oracle name")
    parser.add_argument("--vllm_endpoint", type=str, required=True, help="vLLM endpoint for Qwen-2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="single_objective_output", help="Directory for results")
    parser.add_argument("--sample_method", type=str, default="exp", choices=["exp", "uniform"], help="Sampling method")
    parser.add_argument("--model_path", type=str, default=None, help="Model path for optional DPO training")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for gated models")
    parser.add_argument("--dpo", action="store_true", help="Perform DPO fine tuning after search")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_search(
        n=args.n,
        iterations=args.iterations,
        m=args.m,
        reward=args.reward,
        vllm_endpoint=args.vllm_endpoint,
        output_dir=args.output_dir,
        sample_method=args.sample_method,
        model_path=args.model_path,
        hf_token=args.hf_token,
        dpo=args.dpo,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
