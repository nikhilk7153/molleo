#!/usr/bin/env python
"""
run_multi_iterations.py

Script to run multiple iterations of the RL algorithm, tracking performance over time.
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import json
from pathlib import Path

def run_iteration(pool_file, m, reward, vllm_endpoint, output_file, sample_method='exp'):
    """Run a single iteration using the run_iteration.py script"""
    cmd = [
        'python', 'run_iteration.py',
        '--pool', pool_file,
        '--m', str(m),
        '--reward', reward,
        '--vllm_endpoint', vllm_endpoint,
        '--output', output_file,
        '--sample_method', sample_method
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running iteration: {result.stderr}")
        return None
    
    # Parse RL reward from output
    output = result.stdout
    rl_reward_line = [line for line in output.split('\n') if "RL reward for this iteration" in line]
    if rl_reward_line:
        rl_reward = float(rl_reward_line[0].split(':')[-1].strip())
    else:
        rl_reward = 0.0
    
    return rl_reward

def plot_rewards(rewards, output_file):
    """Plot rewards over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards)+1), rewards, 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RL Reward')
    plt.title('RL Reward over Iterations')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run multiple iterations of the RL algorithm')
    parser.add_argument('--initial_pool', type=str, required=True, help='Path to initial candidate pool pickle file')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to run')
    parser.add_argument('--m', type=int, default=10, help='Number of positive/negative samples per iteration')
    parser.add_argument('--reward', type=str, default='qed', help='Reward function to use')
    parser.add_argument('--vllm_endpoint', type=str, required=True, help='VLLM API endpoint for Qwen-2.5-7b-instruct')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--sample_method', type=str, default='exp', choices=['uniform', 'exp'], 
                      help='Method for sampling from the pool')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize tracking variables
    rewards = []
    current_pool = args.initial_pool
    
    # Save run configuration
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Starting run with {args.iterations} iterations")
    print(f"Initial pool: {args.initial_pool}")
    print(f"Results will be saved to: {run_dir}")
    
    # Run iterations
    for i in range(args.iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {i+1}/{args.iterations}")
        print(f"{'='*50}")
        
        # Define output file for this iteration
        output_file = os.path.join(run_dir, f"pool_iteration_{i+1}.pkl")
        
        # Run iteration
        rl_reward = run_iteration(
            current_pool, 
            args.m, 
            args.reward, 
            args.vllm_endpoint, 
            output_file,
            args.sample_method
        )
        
        if rl_reward is None:
            print(f"Iteration {i+1} failed, stopping run")
            break
        
        rewards.append(rl_reward)
        current_pool = output_file
        
        # Save current rewards
        with open(os.path.join(run_dir, 'rewards.json'), 'w') as f:
            json.dump(rewards, f, indent=2)
        
        # Plot rewards
        plot_rewards(rewards, os.path.join(run_dir, 'rewards_plot.png'))
        
        print(f"Completed iteration {i+1}, current RL reward: {rl_reward:.4f}")
    
    # Final summary
    print("\nRun complete!")
    print(f"Results saved to: {run_dir}")
    if rewards:
        print(f"Final RL reward: {rewards[-1]:.4f}")
        print(f"Best RL reward: {max(rewards):.4f} (iteration {np.argmax(rewards)+1})")
        print(f"Average RL reward: {np.mean(rewards):.4f}")
    
    # Create a symlink to the best pool
    if rewards:
        best_iteration = np.argmax(rewards) + 1
        best_pool = os.path.join(run_dir, f"pool_iteration_{best_iteration}.pkl")
        best_pool_link = os.path.join(run_dir, "best_pool.pkl")
        
        # Create relative symlink
        try:
            best_pool_relative = os.path.relpath(best_pool, run_dir)
            os.symlink(best_pool_relative, best_pool_link)
            print(f"Best pool symlinked to: {best_pool_link}")
        except Exception as e:
            print(f"Could not create symlink: {e}")

if __name__ == "__main__":
    main() 