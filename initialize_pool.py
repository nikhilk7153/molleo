#!/usr/bin/env python
"""
initialize_pool.py

Script to initialize a candidate pool by randomly sampling from ZINC database.
"""

from tdc.generation import MolGen
import numpy as np
from rdkit import Chem
import argparse
import pickle
import os

def initialize_candidate_pool(n, seed=42):
    """
    Initialize a candidate pool of n molecules by randomly sampling from ZINC database.
    
    Args:
        n: Number of molecules to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of n SMILES strings from ZINC database
    """
    print(f"Initializing candidate pool with {n} molecules (seed={seed})...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Load molecules from ZINC database
    print("Loading ZINC database...")
    data = MolGen(name='ZINC')
    all_smiles = data.get_data()['smiles'].tolist()
    print(f"Loaded {len(all_smiles)} molecules from ZINC")
    
    # Randomly sample n molecules
    print(f"Sampling {n} molecules...")
    candidate_pool = np.random.choice(all_smiles, n, replace=False)
    
    # Verify molecules are valid
    print("Verifying molecule validity...")
    valid_pool = []
    for smi in candidate_pool:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_pool.append(smi)
    
    # If we lost some molecules due to validity checking, sample more to reach n
    if len(valid_pool) < n:
        print(f"Found {len(valid_pool)} valid molecules, sampling {n - len(valid_pool)} more...")
        remaining = n - len(valid_pool)
        remaining_smiles = [smi for smi in all_smiles if smi not in valid_pool]
        additional_samples = np.random.choice(remaining_smiles, remaining, replace=False)
        valid_pool.extend(additional_samples)
    
    print(f"Successfully created pool with {len(valid_pool[:n])} molecules")
    return valid_pool[:n]

def main():
    parser = argparse.ArgumentParser(description='Initialize a candidate pool from ZINC database')
    parser.add_argument('--n', type=int, default=1000, help='Number of molecules to sample')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='candidate_pool.pkl', help='Output file to save candidate pool')
    args = parser.parse_args()
    
    # Initialize candidate pool
    candidate_pool = initialize_candidate_pool(args.n, args.seed)
    
    # Save candidate pool to file
    print(f"Saving candidate pool to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(candidate_pool, f)
    
    # Print sample molecules
    print("\nSample molecules from the candidate pool:")
    for i, smi in enumerate(candidate_pool[:5]):
        print(f"{i+1}. {smi}")
    
    print(f"\nCandidate pool of {len(candidate_pool)} molecules saved to {args.output}")

if __name__ == "__main__":
    main() 