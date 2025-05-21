#!/usr/bin/env python
"""
analyze_rewards.py

Script to analyze the individual reward components from molecular generation results.
"""

import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tdc import Oracle
import glob
import json

def calculate_rewards(smiles_list, reward_name='qed'):
    """Calculate rewards for a list of SMILES strings"""
    reward_oracle = Oracle(name=reward_name)
    rewards = []
    
    for smi in smiles_list:
        try:
            reward = reward_oracle(smi)
            rewards.append(reward)
        except:
            # If there's an error, assign minimum reward
            rewards.append(0.0)
    
    return rewards

def calculate_diversity(smiles_list):
    """Calculate molecular diversity of a list of SMILES strings"""
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    
    if len(mols) < 2:
        return 0.0
    
    # Calculate fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols]
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(similarity)
    
    # Diversity is 1 - average similarity
    diversity = 1.0 - (sum(similarities) / len(similarities) if similarities else 0.0)
    return diversity

def analyze_pool(pool_file, previous_pool_file=None, reward_name='qed', alpha=0.5, beta=0.3, gamma=0.2):
    """
    Analyze a pool file and compute individual reward components
    
    Args:
        pool_file: Path to the pool file to analyze
        previous_pool_file: Path to the previous pool file (if any)
        reward_name: Name of the reward function to use
        alpha, beta, gamma: Weights for the reward components
        
    Returns:
        Dictionary containing the reward components
    """
    # Load the pool
    with open(pool_file, 'rb') as f:
        current_pool = pickle.load(f)
    
    # Calculate rewards for current pool
    current_rewards = calculate_rewards(current_pool, reward_name)
    
    # Initialize results
    results = {
        "pool_size": len(current_pool),
        "max_reward": max(current_rewards),
        "mean_reward": np.mean(current_rewards),
        "diversity": calculate_diversity(current_pool),
    }
    
    # If we have a previous pool, compute improvement metrics
    if previous_pool_file:
        with open(previous_pool_file, 'rb') as f:
            previous_pool = pickle.load(f)
        
        # Find new molecules
        new_molecules = [smi for smi in current_pool if smi not in previous_pool]
        
        if new_molecules:
            # Calculate rewards for previous pool and new molecules
            previous_rewards = calculate_rewards(previous_pool, reward_name)
            new_rewards = calculate_rewards(new_molecules, reward_name)
            
            # Calculate components
            max_improvement = max(0, max(new_rewards) - max(previous_rewards))
            mean_improvement = max(0, np.mean(new_rewards) - np.mean(previous_rewards))
            diversity_value = calculate_diversity(new_molecules)
            
            # Calculate total reward
            total_reward = (alpha * max_improvement + 
                           beta * mean_improvement + 
                           gamma * diversity_value)
            
            # Add to results
            results.update({
                "new_molecules": len(new_molecules),
                "max_improvement": max_improvement,
                "mean_improvement": mean_improvement,
                "diversity_of_new": diversity_value,
                "total_reward": total_reward,
                "max_reward_new": max(new_rewards),
                "mean_reward_new": np.mean(new_rewards),
            })
        else:
            results["new_molecules"] = 0
    
    return results

def analyze_run_directory(run_dir, reward_name='qed', alpha=0.5, beta=0.3, gamma=0.2):
    """Analyze all pool files in a run directory"""
    # Find all pool files
    pool_files = sorted(glob.glob(os.path.join(run_dir, "pool_iteration_*.pkl")))
    
    if not pool_files:
        print(f"No pool files found in {run_dir}")
        return
    
    print(f"Found {len(pool_files)} pool files in {run_dir}")
    
    # Analyze each pool file
    results = []
    previous_pool = None
    
    for i, pool_file in enumerate(pool_files):
        print(f"Analyzing {os.path.basename(pool_file)}...")
        result = analyze_pool(pool_file, previous_pool, reward_name, alpha, beta, gamma)
        results.append(result)
        previous_pool = pool_file
    
    # Save results
    results_file = os.path.join(run_dir, "reward_analysis.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(results, run_dir)
    
    return results

def plot_results(results, output_dir):
    """Plot the reward components"""
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Iterations
    iterations = list(range(1, len(results) + 1))
    
    # Plot pool size
    axes[0, 0].plot(iterations, [r["pool_size"] for r in results], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Pool Size')
    axes[0, 0].set_title('Molecule Pool Size')
    axes[0, 0].grid(True)
    
    # Plot max and mean rewards
    axes[0, 1].plot(iterations, [r["max_reward"] for r in results], 'o-', linewidth=2, label='Max Reward')
    axes[0, 1].plot(iterations, [r["mean_reward"] for r in results], 'o-', linewidth=2, label='Mean Reward')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Reward Value')
    axes[0, 1].set_title('Reward Values')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot diversity
    axes[1, 0].plot(iterations, [r["diversity"] for r in results], 'o-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Diversity')
    axes[1, 0].set_title('Molecular Diversity')
    axes[1, 0].grid(True)
    
    # Plot reward components if available
    if "total_reward" in results[1]:  # Check if we have component data (not available for first iteration)
        iter_with_components = iterations[1:]  # Skip first iteration
        component_results = results[1:]  # Skip first iteration
        
        axes[1, 1].plot(iter_with_components, [r["max_improvement"] for r in component_results], 'o-', linewidth=2, label='Max Improvement')
        axes[1, 1].plot(iter_with_components, [r["mean_improvement"] for r in component_results], 'o-', linewidth=2, label='Mean Improvement')
        axes[1, 1].plot(iter_with_components, [r["diversity_of_new"] for r in component_results], 'o-', linewidth=2, label='Diversity')
        axes[1, 1].plot(iter_with_components, [r["total_reward"] for r in component_results], 'o-', linewidth=2, label='Total Reward')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Reward Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_components.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze reward components from molecular generation results')
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the run directory')
    parser.add_argument('--reward', type=str, default='qed', help='Reward function used')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for max improvement')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for mean improvement')
    parser.add_argument('--gamma', type=float, default=0.2, help='Weight for diversity')
    args = parser.parse_args()
    
    print(f"Analyzing run directory: {args.run_dir}")
    print(f"Reward function: {args.reward}")
    print(f"Component weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")
    
    # Analyze the run directory
    results = analyze_run_directory(args.run_dir, args.reward, args.alpha, args.beta, args.gamma)
    
    if results:
        print("\nResults summary:")
        for i, result in enumerate(results):
            print(f"\nIteration {i+1}:")
            print(f"  Pool size: {result['pool_size']}")
            print(f"  Max reward: {result['max_reward']:.4f}")
            print(f"  Mean reward: {result['mean_reward']:.4f}")
            print(f"  Diversity: {result['diversity']:.4f}")
            
            if i > 0 and "total_reward" in result:
                print(f"  New molecules: {result['new_molecules']}")
                print(f"  Max improvement: {result['max_improvement']:.4f}")
                print(f"  Mean improvement: {result['mean_improvement']:.4f}")
                print(f"  Diversity of new: {result['diversity_of_new']:.4f}")
                print(f"  Total reward: {result['total_reward']:.4f}")
    
    print(f"\nResults saved to {os.path.join(args.run_dir, 'reward_analysis.json')}")
    print(f"Plots saved to {os.path.join(args.run_dir, 'reward_components.png')}")

if __name__ == "__main__":
    main() 