#!/usr/bin/env python
"""
run_iteration.py

Script to run a single iteration of the RL algorithm, which includes:
1. Sampling from the candidate pool
2. Scoring with reward oracle
3. Generating new samples using LLM
4. Evaluating new samples and updating the pool
"""

import argparse
import pickle
import os
import numpy as np
import re
import requests
import json
from rdkit import Chem
from tdc import Oracle
from datetime import datetime

def load_candidate_pool(filename):
    """Load candidate pool from pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

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

def sample_from_pool(candidate_pool, rewards, m, sample_method='exp'):
    """
    Sample molecules from the candidate pool.
    
    Args:
        candidate_pool: List of SMILES strings
        rewards: List of reward values for each molecule
        m: Number of samples to take for positive and negative
        sample_method: 'exp' for exponential weighting, 'uniform' for uniform sampling
        
    Returns:
        positive_samples: List of (SMILES, reward) pairs for positive examples
        negative_samples: List of (SMILES, reward) pairs for negative examples
    """
    # Create (SMILES, reward) pairs
    candidates_with_rewards = list(zip(candidate_pool, rewards))
    
    if sample_method == 'uniform':
        # Uniform sampling - simply sort and take top/bottom m
        sorted_candidates = sorted(candidates_with_rewards, key=lambda x: x[1], reverse=True)
        positive_samples = sorted_candidates[:m]
        negative_samples = sorted_candidates[-m:]
    
    elif sample_method == 'exp':
        # Exponential weighting
        # For positive samples: p(i) ~ exp(reward(i))
        positive_weights = np.exp([r for _, r in candidates_with_rewards])
        positive_weights = positive_weights / np.sum(positive_weights)
        
        # For negative samples: p(i) ~ exp(-reward(i))
        negative_weights = np.exp([-r for _, r in candidates_with_rewards])
        negative_weights = negative_weights / np.sum(negative_weights)
        
        # Sample indices
        positive_indices = np.random.choice(
            len(candidates_with_rewards), 
            size=m, 
            replace=False, 
            p=positive_weights
        )
        negative_indices = np.random.choice(
            len(candidates_with_rewards), 
            size=m, 
            replace=False, 
            p=negative_weights
        )
        
        # Get samples
        positive_samples = [candidates_with_rewards[i] for i in positive_indices]
        negative_samples = [candidates_with_rewards[i] for i in negative_indices]
    
    else:
        raise ValueError(f"Unknown sampling method: {sample_method}")
    
    return positive_samples, negative_samples

def prompt_llm(positive_samples, negative_samples, vllm_endpoint, m):
    """
    Prompt LLMs hosted on VLLM to generate new molecules.
    Supports various model formats including Qwen, Mistral, and Llama families.
    
    Args:
        positive_samples: List of (SMILES, reward) pairs for positive examples
        negative_samples: List of (SMILES, reward) pairs for negative examples
        vllm_endpoint: URL of the VLLM server
        m: Number of new samples to generate
        
    Returns:
        List of generated SMILES strings
    """
    # Format samples for the prompt
    positive_formatted = "\n".join([f"SMILES: {smi}, Reward: {reward:.4f}" 
                                  for smi, reward in positive_samples])
    negative_formatted = "\n".join([f"SMILES: {smi}, Reward: {reward:.4f}" 
                                  for smi, reward in negative_samples])
    
    # Create a common content part that will be used in all formats
    content = f"""Here are {len(positive_samples)} positive samples with high rewards:
{positive_formatted}

Here are {len(negative_samples)} negative samples with low rewards:
{negative_formatted}

Please analyze the results, and output {m} new SMILES strings for molecules with rewards better than the positive samples. The new samples should be diversified and have valid chemical structures.

Please output your answer in the following format:
1. SMILES: [your generated SMILES]
2. SMILES: [your generated SMILES]
...
{m}. SMILES: [your generated SMILES]

Only output the SMILES strings, do not include any explanation or additional text.
"""
    
    # Try to detect which model format to use based on the first response
    # First try with Qwen format
    try:
        # Format for Qwen models
        qwen_prompt = f"""<|im_start|>system
You are a black-box reward optimizer for molecular generation.
<|im_end|>
<|im_start|>user
{content}
<|im_end|>
<|im_start|>assistant
"""
        
        # Call the API with Qwen format first
        payload = {
            "prompt": qwen_prompt,
            "temperature": 0.7,
            "max_tokens": 2000,
            "stop": ["<|im_end|>"]
        }
        
        response = requests.post(vllm_endpoint, json=payload)
        response.raise_for_status()
        
        # Extract the generated text
        response_json = response.json()
        response_text = get_response_text(response_json)
        smiles_list = extract_smiles(response_text)
        
        if smiles_list:
            return validate_smiles(smiles_list)
            
    except Exception as e:
        print(f"Qwen format attempt failed: {e}")
    
    # Try other formats if Qwen format fails
    try:
        # Format for Mistral/Llama models (ChatML format)
        chatml_prompt = f"""<s>[INST]
You are a black-box reward optimizer for molecular generation.

{content}
[/INST]
"""
        
        payload = {
            "prompt": chatml_prompt,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(vllm_endpoint, json=payload)
        response.raise_for_status()
        
        # Extract the generated text
        response_json = response.json()
        response_text = get_response_text(response_json)
        smiles_list = extract_smiles(response_text)
        
        if smiles_list:
            return validate_smiles(smiles_list)
            
    except Exception as e:
        print(f"ChatML format attempt failed: {e}")
    
    # Try with plain text format as last resort
    try:
        # Plain text format
        plain_prompt = f"""You are a black-box reward optimizer for molecular generation.

{content}
"""
        
        payload = {
            "prompt": plain_prompt,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(vllm_endpoint, json=payload)
        response.raise_for_status()
        
        # Extract the generated text
        response_json = response.json()
        response_text = get_response_text(response_json)
        smiles_list = extract_smiles(response_text)
        
        return validate_smiles(smiles_list)
            
    except Exception as e:
        print(f"Error calling VLLM API with all formats: {e}")
        # Return empty list if all API calls fail
        return []

def get_response_text(response_json):
    """Extract text from VLLM response which can have different formats"""
    if isinstance(response_json, list) and len(response_json) > 0:
        return response_json[0].get("generated_text", "")
    elif "generated_text" in response_json:
        return response_json["generated_text"]
    else:
        print(f"Unexpected response format: {response_json}")
        return ""

def extract_smiles(response_text):
    """Extract SMILES strings from response text"""
    import re
    return re.findall(r'\d+\.\s*SMILES:\s*([^\n]+)', response_text)

def validate_smiles(smiles_list):
    """Validate and canonicalize SMILES strings"""
    valid_smiles = []
    for smi in smiles_list:
        smi = smi.strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol))  # Canonicalize
    return valid_smiles

def calculate_diversity(smiles_list):
    """Calculate molecular diversity of a list of SMILES strings"""
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    
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

def calculate_rl_reward(new_samples_rewards, positive_samples, diversity_value, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Calculate the RL reward for the generated samples.
    
    Args:
        new_samples_rewards: List of reward values for new samples
        positive_samples: List of (SMILES, reward) pairs for positive examples
        diversity_value: Diversity measure of new samples
        alpha, beta, gamma: Weights for the different components
        
    Returns:
        Total reward value
    """
    if not new_samples_rewards:
        return 0.0
    
    # Get rewards from positive samples
    positive_rewards = [r for _, r in positive_samples]
    
    # Calculate components
    reward_max_improvement = max(0, max(new_samples_rewards, default=0) - max(positive_rewards, default=0))
    reward_mean_improvement = max(0, np.mean(new_samples_rewards) - np.mean(positive_rewards) if new_samples_rewards else 0)
    
    # Calculate total reward
    total_reward = (alpha * reward_max_improvement + 
                   beta * reward_mean_improvement + 
                   gamma * diversity_value)
    
    return total_reward

def main():
    parser = argparse.ArgumentParser(description='Run a single iteration of the RL algorithm')
    parser.add_argument('--pool', type=str, required=True, help='Path to candidate pool pickle file')
    parser.add_argument('--m', type=int, default=10, help='Number of positive/negative samples')
    parser.add_argument('--reward', type=str, default='qed', help='Reward function to use')
    parser.add_argument('--vllm_endpoint', type=str, required=True, help='VLLM API endpoint for Qwen-2.5-7b-instruct')
    parser.add_argument('--output', type=str, default=None, help='Output file for updated pool')
    parser.add_argument('--sample_method', type=str, default='exp', choices=['uniform', 'exp'], 
                        help='Method for sampling from the pool')
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"updated_pool_{timestamp}.pkl"
    
    # Load candidate pool
    print(f"Loading candidate pool from {args.pool}...")
    candidate_pool = load_candidate_pool(args.pool)
    print(f"Loaded {len(candidate_pool)} molecules")
    
    # Calculate rewards for all molecules in the pool
    print(f"Calculating {args.reward} rewards for all molecules...")
    rewards = calculate_rewards(candidate_pool, args.reward)
    
    # Sample positive and negative examples
    print(f"Sampling {args.m} positive and {args.m} negative examples using {args.sample_method} method...")
    positive_samples, negative_samples = sample_from_pool(
        candidate_pool, rewards, args.m, args.sample_method
    )
    
    # Print samples
    print("\nPositive samples:")
    for i, (smi, reward) in enumerate(positive_samples):
        print(f"{i+1}. SMILES: {smi}, Reward: {reward:.4f}")
    
    print("\nNegative samples:")
    for i, (smi, reward) in enumerate(negative_samples):
        print(f"{i+1}. SMILES: {smi}, Reward: {reward:.4f}")
    
    # Generate new samples using Qwen-2.5-7b on VLLM
    print(f"\nGenerating {args.m} new samples using Qwen-2.5-7b-instruct on VLLM...")
    new_samples = prompt_llm(positive_samples, negative_samples, args.vllm_endpoint, args.m)
    print(f"Generated {len(new_samples)} valid new samples")
    
    # Calculate rewards for new samples
    print("Calculating rewards for new samples...")
    new_samples_rewards = calculate_rewards(new_samples, args.reward)
    
    # Calculate diversity
    print("Calculating diversity of new samples...")
    diversity_value = calculate_diversity(new_samples)
    print(f"Diversity: {diversity_value:.4f}")
    
    # Print new samples with rewards
    print("\nNew samples:")
    for i, (smi, reward) in enumerate(zip(new_samples, new_samples_rewards)):
        print(f"{i+1}. SMILES: {smi}, Reward: {reward:.4f}")
    
    # Calculate RL reward
    rl_reward = calculate_rl_reward(new_samples_rewards, positive_samples, diversity_value)
    print(f"\nRL reward for this iteration: {rl_reward:.4f}")
    
    # Update candidate pool with new samples
    print("Updating candidate pool...")
    updated_pool = list(candidate_pool)
    for smi in new_samples:
        if smi not in updated_pool:
            updated_pool.append(smi)
    
    # Save updated pool
    print(f"Saving updated pool ({len(updated_pool)} molecules) to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(updated_pool, f)
    
    print(f"Success! Updated pool saved to {args.output}")

if __name__ == "__main__":
    main() 