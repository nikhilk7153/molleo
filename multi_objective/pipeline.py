#!/usr/bin/env python
"""Pipeline for multi-objective molecular optimization using Qwen-2.5-7B-Instruct.

This script follows the evolutionary search procedure from
"Efficient Evolutionary Search Over Chemical Space with Large Language Models".
It extends the single-objective pipeline to support multiple reward objectives
both for maximization and minimization. Direct Preference Optimization (DPO)
training is optional.
"""

import argparse
import os
import pickle

from initialize_pool import initialize_candidate_pool
from run_iteration import (
    sample_from_pool,
    prompt_llm,
    calculate_diversity,
    calculate_rl_reward,
)

from tdc import Oracle

# Optional imports only required when DPO is requested
try:
    from rl_train import sample_trajectories, train_with_dpo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - handled gracefully
    sample_trajectories = train_with_dpo = None


def calculate_multi_rewards(smiles_list, max_obj, min_obj):
    """Aggregate rewards over multiple objectives."""
    max_oracles = [Oracle(name=o) for o in max_obj]
    min_oracles = [Oracle(name=o) for o in min_obj]

    rewards = []
    for smi in smiles_list:
        total = 0.0
        for eva in max_oracles:
            try:
                total += eva(smi)
            except Exception:
                pass
        for eva in min_oracles:
            try:
                val = eva(smi)
                if eva.name.lower() == "sa":
                    total += 1 - ((val - 1) / 9)
                else:
                    total += 1 - val
            except Exception:
                pass
        rewards.append(total)
    return rewards


def create_preference_pairs_multi(trajectories, max_obj, min_obj):
    """Create preference pairs based on aggregated multi-objective rewards."""
    if not trajectories:
        return [], []

    smiles = [t["smiles"] for t in trajectories]
    rewards = calculate_multi_rewards(smiles, max_obj, min_obj)

    for i, tr in enumerate(trajectories):
        tr["reward"] = rewards[i]

    sorted_traj = sorted(trajectories, key=lambda x: x["reward"], reverse=True)
    midpoint = len(sorted_traj) // 2
    preferred = []
    rejected = []
    for i in range(midpoint):
        preferred.append(sorted_traj[i]["full_text"])
        rejected.append(sorted_traj[-(i + 1)]["full_text"])
    return preferred, rejected


def run_search(
    n,
    iterations,
    m,
    max_obj,
    min_obj,
    vllm_endpoint,
    output_dir,
    sample_method="exp",
    model_path=None,
    hf_token=None,
    dpo=False,
    seed=42,
):
    """Run the multi-objective evolutionary search."""
    os.makedirs(output_dir, exist_ok=True)

    candidate_pool = initialize_candidate_pool(n, seed)
    with open(os.path.join(output_dir, "pool_iteration_0.pkl"), "wb") as f:
        pickle.dump(candidate_pool, f)

    all_pref = []
    all_rej = []

    for iteration in range(1, iterations + 1):
        print(f"\nIteration {iteration}/{iterations}")
        rewards = calculate_multi_rewards(candidate_pool, max_obj, min_obj)
        pos, neg = sample_from_pool(candidate_pool, rewards, m, sample_method)

        new_samples = prompt_llm(pos, neg, vllm_endpoint, m)
        new_rewards = calculate_multi_rewards(new_samples, max_obj, min_obj)
        diversity = calculate_diversity(new_samples)
        rl_reward = calculate_rl_reward(new_rewards, pos, diversity)
        print(f"RL reward: {rl_reward:.4f}")

        for smi in new_samples:
            if smi not in candidate_pool:
                candidate_pool.append(smi)

        with open(
            os.path.join(output_dir, f"pool_iteration_{iteration}.pkl"), "wb"
        ) as f:
            pickle.dump(candidate_pool, f)

        if dpo and sample_trajectories is not None:
            pos_fmt = "\n".join([f"SMILES: {s}, Reward: {r:.4f}" for s, r in pos])
            neg_fmt = "\n".join([f"SMILES: {s}, Reward: {r:.4f}" for s, r in neg])
            content = (
                f"Here are {m} positive samples with high rewards:\n{pos_fmt}\n\n"
                f"Here are {m} negative samples with low rewards:\n{neg_fmt}\n\n"
                f"Please analyze the results, and output {m} new SMILES strings "
                f"with rewards better than the positive samples. The new samples should be diversified."
            )
            prompt = (
                "<|im_start|>system\nYou are a black-box reward optimizer for molecular generation."\
                "<|im_end|>\n<|im_start|>user\n" + content + "\n<|im_end|>\n<|im_start|>assistant"
            )
            trajs = sample_trajectories(None, None, prompt, vllm_endpoint, num_samples=m)
            pref, rej = create_preference_pairs_multi(trajs, max_obj, min_obj)
            all_pref.extend(pref)
            all_rej.extend(rej)

    if dpo and all_pref and model_path is not None and train_with_dpo is not None:
        print("\nStarting DPO fine tuning...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        train_with_dpo(model, tokenizer, all_pref, all_rej, os.path.join(output_dir, "dpo_model"))

    print("\nSearch completed. Results stored in", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Multi-objective molecular optimization pipeline")
    parser.add_argument("--n", type=int, default=1000, help="Initial pool size")
    parser.add_argument("--iterations", type=int, default=3, help="Number of optimization iterations")
    parser.add_argument("--m", type=int, default=10, help="Number of positive/negative samples per iteration")
    parser.add_argument("--max_obj", nargs="+", default=["jnk3", "qed"], help="Objectives to maximize")
    parser.add_argument("--min_obj", nargs="+", default=["sa"], help="Objectives to minimize")
    parser.add_argument("--vllm_endpoint", type=str, required=True, help="vLLM endpoint for Qwen-2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="multi_objective_output", help="Directory for results")
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
        max_obj=args.max_obj,
        min_obj=args.min_obj,
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
