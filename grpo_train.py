#!/usr/bin/env python
"""Generalized RL Preference Optimization training script.

This script combines Reinforcement Learning with Variance Reduction (RLVR)
and Direct Preference Optimization (DPO). It first runs an RLVR loop to
collect prompt/response pairs with rewards and fine-tunes the model with a
value head. The resulting data is then converted to preference pairs and
used for a final DPO stage.
"""

import argparse
import os
import pickle
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvr_train import RLDataset, collate_fn, train_rlvr
from rl_train import train_with_dpo
from run_iteration import (
    calculate_rewards,
    calculate_diversity,
    calculate_rl_reward,
    sample_from_pool,
    prompt_llm,
)


def run_rlvr_loop(
    pool: List[str],
    iterations: int,
    m: int,
    reward_name: str,
    endpoint: str,
) -> Tuple[List[str], List[str], List[float], List[str]]:
    """Run iterative generation to collect data for RLVR."""
    prompts: List[str] = []
    responses: List[str] = []
    rewards: List[float] = []

    candidate_pool = pool
    for _ in range(iterations):
        cand_rewards = calculate_rewards(candidate_pool, reward_name)
        pos, neg = sample_from_pool(candidate_pool, cand_rewards, m)
        new_samples = prompt_llm(pos, neg, endpoint, m)
        new_rewards = calculate_rewards(new_samples, reward_name)
        diversity = calculate_diversity(new_samples)
        r = calculate_rl_reward(new_rewards, pos, diversity)

        content = "\n".join([f"SMILES: {s}" for s in new_samples]) + "\n"
        prompt = "Generate new molecules:" + "\n".join(
            [f"SMILES: {s} ({v:.2f})" for s, v in pos]
        ) + "\n"
        prompts.append(prompt)
        responses.append(content)
        rewards.append(r)
        candidate_pool.extend([s for s in new_samples if s not in candidate_pool])
    return prompts, responses, rewards, candidate_pool


def make_preference_pairs(
    prompts: List[str],
    responses: List[str],
    rewards: List[float],
) -> Tuple[List[str], List[str]]:
    """Create preference pairs from RLVR data."""
    triples = sorted(zip(prompts, responses, rewards), key=lambda x: x[2], reverse=True)
    mid = len(triples) // 2
    preferred = [p + r for p, r, _ in triples[:mid]]
    rejected = [p + r for p, r, _ in triples[-mid:]]
    return preferred, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training (RLVR + DPO)")
    parser.add_argument("--pool", required=True, help="Candidate pool pickle file")
    parser.add_argument("--model", required=True, help="Base model path or HF id")
    parser.add_argument("--vllm_endpoint", required=True, help="VLLM generation endpoint")
    parser.add_argument("--output_dir", default="grpo_model", help="Directory for results")
    parser.add_argument("--iterations", type=int, default=3, help="RL iterations")
    parser.add_argument("--m", type=int, default=10, help="Number of pos/neg samples")
    parser.add_argument("--reward", type=str, default="qed", help="Reward oracle name")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional HF token")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.model, token=args.hf_token)
    value_head = torch.nn.Linear(model.config.hidden_size, 1)

    with open(args.pool, "rb") as f:
        pool = pickle.load(f)

    prompts, responses, rewards, _ = run_rlvr_loop(
        pool, args.iterations, args.m, args.reward, args.vllm_endpoint
    )

    dataset = RLDataset(prompts, responses, rewards, tokenizer)
    train_rlvr(
        model,
        tokenizer,
        dataset,
        value_head,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    pref, rej = make_preference_pairs(prompts, responses, rewards)
    train_with_dpo(
        model,
        tokenizer,
        pref,
        rej,
        os.path.join(args.output_dir, "dpo"),
    )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(value_head.state_dict(), os.path.join(args.output_dir, "value_head.pt"))


if __name__ == "__main__":
    main()
