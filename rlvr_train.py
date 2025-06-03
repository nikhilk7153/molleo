#!/usr/bin/env python
"""Reinforcement Learning with Variance Reduction (RLVR) training script.

This script fine-tunes a causal language model with a simple actor-critic
objective. The actor (the language model) generates molecules based on prompts
from the evolutionary search. A small value head predicts expected reward and
serves as a baseline for variance reduction.

The reward combines the improvement in maximum and mean rewards of the generated
molecules with their diversity, mirroring the single-objective pipeline.
"""
import argparse
import os
import pickle
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_iteration import (
    calculate_rewards,
    calculate_diversity,
    calculate_rl_reward,
    sample_from_pool,
    prompt_llm,
)


class RLDataset(Dataset):
    """Simple dataset of prompt/response pairs with rewards."""

    def __init__(self, prompts: List[str], responses: List[str], rewards: List[float], tokenizer):
        self.inputs = []
        self.attn = []
        self.response_masks = []
        for p, r in zip(prompts, responses):
            enc = tokenizer(p + r, return_tensors="pt")
            self.inputs.append(enc["input_ids"].squeeze())
            self.attn.append(enc["attention_mask"].squeeze())
            resp_len = len(tokenizer(r, add_special_tokens=False))
            mask = torch.zeros_like(self.inputs[-1], dtype=torch.bool)
            mask[-resp_len:] = True
            self.response_masks.append(mask)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": self.attn[idx],
            "response_mask": self.response_masks[idx],
            "reward": self.rewards[idx],
        }


def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attn = []
    resp_mask = []
    rewards = []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)]))
        attn.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        resp_mask.append(torch.cat([item["response_mask"], torch.zeros(pad_len, dtype=torch.bool)]))
        rewards.append(item["reward"])
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "response_mask": torch.stack(resp_mask),
        "reward": torch.stack(rewards),
    }


def train_rlvr(model, tokenizer, dataset, value_head, lr=5e-6, epochs=1, device="cpu"):
    model.to(device)
    value_head.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(value_head.parameters()), lr=lr)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    for _ in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            mask = batch["response_mask"].to(device)
            reward = batch["reward"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Gather log-probs for generated tokens only
            resp_indices = mask.nonzero(as_tuple=True)
            token_logprob = log_probs[resp_indices][torch.arange(mask.sum()), input_ids[resp_indices]]
            logprob = token_logprob.mean()

            # Baseline from value head using last hidden state
            last_hidden = outputs.hidden_states[-1][resp_indices[0], resp_indices[1]]
            baseline = value_head(last_hidden).mean()

            advantage = reward - baseline.detach()
            policy_loss = -(advantage * logprob)
            value_loss = torch.nn.functional.mse_loss(baseline, reward)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL training with variance reduction")
    parser.add_argument("--pool", type=str, required=True, help="Candidate pool pickle file")
    parser.add_argument("--model", type=str, required=True, help="Base model path or identifier")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace auth token")
    parser.add_argument("--reward", type=str, default="qed", help="Reward oracle")
    parser.add_argument("--vllm_endpoint", type=str, required=True, help="vLLM endpoint for generation")
    parser.add_argument("--m", type=int, default=10, help="Number of positive/negative samples")
    parser.add_argument("--iterations", type=int, default=3, help="Number of RL iterations")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.model, token=args.hf_token)
    value_head = nn.Linear(model.config.hidden_size, 1)

    with open(args.pool, "rb") as f:
        pool = pickle.load(f)

    prompts = []
    responses = []
    rewards = []

    candidate_pool = pool
    for _ in range(args.iterations):
        cand_rewards = calculate_rewards(candidate_pool, args.reward)
        pos, neg = sample_from_pool(candidate_pool, cand_rewards, args.m)
        new_samples = prompt_llm(pos, neg, args.vllm_endpoint, args.m)
        new_rewards = calculate_rewards(new_samples, args.reward)
        diversity = calculate_diversity(new_samples)
        r = calculate_rl_reward(new_rewards, pos, diversity)

        content = "\n".join([f"SMILES: {s}" for s in new_samples]) + "\n"
        prompt = "Generate new molecules:"\
            + "\n".join([f"SMILES: {s} ({v:.2f})" for s, v in pos]) + "\n"
        prompts.append(prompt)
        responses.append(content)
        rewards.append(r)
        candidate_pool.extend([s for s in new_samples if s not in candidate_pool])

    dataset = RLDataset(prompts, responses, rewards, tokenizer)
    train_rlvr(model, tokenizer, dataset, value_head, device="cuda" if torch.cuda.is_available() else "cpu")

    model.save_pretrained("rlvr_model")
    tokenizer.save_pretrained("rlvr_model")
    torch.save(value_head.state_dict(), os.path.join("rlvr_model", "value_head.pt"))
