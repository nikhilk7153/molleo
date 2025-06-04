#!/usr/bin/env python
"""Train a model using the verl PPO trainer on a molecular dataset."""

import argparse
import os
import shutil

from initialize_pool import initialize_candidate_pool
from verl_utils import make_verl_dataset, run_verl_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train using verl PPO")
    parser.add_argument("--model", required=True, help="Base model path or HF identifier")
    parser.add_argument("--output_dir", default="verl_output", help="Directory for datasets and results")
    parser.add_argument("--n", type=int, default=1000, help="Initial pool size")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token if needed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize a pool of molecules and create dataset
    pool = initialize_candidate_pool(args.n)
    train_file = os.path.join(args.output_dir, "train.parquet")
    val_file = os.path.join(args.output_dir, "val.parquet")
    make_verl_dataset(pool, train_file)
    shutil.copy(train_file, val_file)

    reward_script = os.path.join(args.output_dir, "verl_reward.py")
    if not os.path.exists(reward_script):
        shutil.copy("verl_reward.py", reward_script)

    extra = []
    if args.hf_token:
        extra.append(f"actor_rollout_ref.model.token={args.hf_token}")
        extra.append(f"critic.model.token={args.hf_token}")

    run_verl_ppo(args.model, train_file, val_file, reward_script, args.output_dir, extra)


if __name__ == "__main__":
    main()
