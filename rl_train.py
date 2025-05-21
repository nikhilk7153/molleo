#!/usr/bin/env python
"""
rl_train.py

Script to perform RL training using reward metrics and DPO (Direct Preference Optimization)
"""

import argparse
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import requests
from rdkit import Chem
from tdc import Oracle
from datetime import datetime
from tqdm import tqdm
import random
import json
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import re

class MoleculeGenerationDataset(Dataset):
    """Dataset for training with preference pairs"""
    def __init__(self, tokenizer, preferred_texts, rejected_texts, max_length=512):
        self.tokenizer = tokenizer
        self.preferred_texts = preferred_texts
        self.rejected_texts = rejected_texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.preferred_texts)
    
    def __getitem__(self, idx):
        preferred_text = self.preferred_texts[idx]
        rejected_text = self.rejected_texts[idx]
        
        # Tokenize inputs
        preferred_tokens = self.tokenizer(
            preferred_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "preferred_input_ids": preferred_tokens["input_ids"].squeeze(),
            "preferred_attention_mask": preferred_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze()
        }

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

def generate_prompt(positive_samples, negative_samples, request_type=None):
    """
    Generate a prompt for the model based on positive and negative examples.
    Different request_type values will generate different prompts for exploration.
    """
    # Format samples for the prompt
    # Only use 3-5 examples to avoid overwhelming the model
    pos_samples = positive_samples[:5]
    neg_samples = negative_samples[:5]
    
    # Find the highest reward in positive samples
    highest_reward = max([reward for _, reward in pos_samples])
    
    positive_formatted = "\n".join([f"SMILES: {smi}, Reward: {reward:.4f}" 
                                  for smi, reward in pos_samples])
    negative_formatted = "\n".join([f"SMILES: {smi}, Reward: {reward:.4f}" 
                                  for smi, reward in neg_samples])
    
    # Create different instructions based on request_type for exploration
    if request_type == "optimize_qed":
        instruction = f"Generate a new SMILES string for a molecule with optimal QED (drug-likeness) properties. Try to exceed the highest QED score of {highest_reward:.4f}."
    elif request_type == "optimize_diversity":
        instruction = "Generate a new SMILES string for a molecule that is structurally diverse but still maintains high reward properties."
    elif request_type == "optimize_synthetic":
        instruction = "Generate a new SMILES string for a molecule that would be easier to synthesize while maintaining similar reward properties."
    else:
        instruction = f"Generate a new SMILES string for a molecule that has better properties than the positive samples. Try to achieve a QED score higher than {highest_reward:.4f}."
    
    # Format for Llama-3.1-8B-Instruct
    prompt = f"""<s>[INST]
You are a computational chemist specializing in drug design. I'll show you some molecules with their QED drug-likeness scores.

Positive examples (high QED molecules):
{positive_formatted}

Negative examples (low QED molecules):
{negative_formatted}

{instruction}

Important rules for the SMILES:
1. Use only valid chemical atoms and bonds
2. Check that rings are properly closed
3. Make sure the molecule is drug-like (follow Lipinski's rules)
4. Balance charges appropriately

Please respond with only the SMILES string of your designed molecule:
SMILES: 
[/INST]
"""
    
    return prompt

def extract_smiles(text):
    """Extract SMILES strings from generated text with improved pattern matching"""
    # Print raw text for debugging
    print(f"Raw generated text:\n{text}")
    
    # Try multiple patterns to extract SMILES
    patterns = [
        r'SMILES:\s*([^\s\n]+)',  # Standard format: "SMILES: CCC..."
        r'SMILES\s+string:\s*([^\s\n]+)',  # Alt format: "SMILES string: CCC..."
        r'([CN][^"\s\n]+)',  # Raw SMILES often start with C or N
        r'([COc][A-Za-z0-9@\[\]\(\)\.=#+-]+)'  # More permissive pattern
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                # Validate with RDKit
                mol = Chem.MolFromSmiles(match)
                if mol is not None:
                    return match
    
    return None

def call_api_with_retry(url, payload, max_retries=3):
    """Call the API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API call attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # Wait before retrying

def sample_molecules_from_api(prompt, vllm_endpoint=None, temperature=0.8, max_tokens=256, n_samples=5):
    """Sample molecules from API endpoint with explicit error handling"""
    if not vllm_endpoint:
        print("No API endpoint provided. Using direct model generation.")
        return None
    
    try:
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = call_api_with_retry(vllm_endpoint, payload)
        
        if response and "generated_text" in response:
            generated_text = response["generated_text"]
            smiles = extract_smiles(generated_text)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return Chem.MolToSmiles(mol)  # Return canonicalized SMILES
        
        print("Failed to extract valid SMILES from API response")
        return None
    
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

def sample_trajectories(model, tokenizer, prompt, vllm_endpoint=None, num_samples=10, temperature_range=[0.5, 0.7, 0.9, 1.1]):
    """
    Sample multiple generation trajectories by varying temperature
    Both supports direct model inference and API calls
    """
    trajectories = []
    
    if vllm_endpoint:
        # Use API endpoint for generation
        for temp in temperature_range:
            print(f"Sampling with temperature {temp}...")
            for _ in range(max(1, num_samples // len(temperature_range))):
                try:
                    payload = {
                        "prompt": prompt,
                        "temperature": temp,
                        "max_tokens": 256
                    }
                    
                    response = requests.post(vllm_endpoint, json=payload)
                    response.raise_for_status()
                    
                    response_json = response.json()
                    if "generated_text" in response_json:
                        generated_text = response_json["generated_text"]
                        
                        # Try to extract SMILES
                        smiles = extract_smiles(generated_text)
                        if smiles:
                            # Validate SMILES
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                canonical_smiles = Chem.MolToSmiles(mol)
                                trajectories.append({
                                    "smiles": canonical_smiles,
                                    "temperature": temp,
                                    "full_text": generated_text
                                })
                                print(f"Successfully extracted SMILES: {canonical_smiles}")
                    else:
                        print(f"No 'generated_text' in response: {response_json}")
                
                except Exception as e:
                    print(f"Error during API sampling: {e}")
    else:
        # Use direct model inference
        for temp in temperature_range:
            print(f"Sampling with temperature {temp}...")
            # Encode the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate with current parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    num_return_sequences=max(1, num_samples // len(temperature_range)),
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and extract generated molecules
            for output in outputs:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                
                # Try to extract SMILES
                smiles = extract_smiles(generated_text)
                if smiles:
                    # Validate SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        trajectories.append({
                            "smiles": canonical_smiles,
                            "temperature": temp,
                            "full_text": generated_text
                        })
                        print(f"Successfully extracted SMILES: {canonical_smiles}")
    
    return trajectories

def create_preference_pairs(trajectories, reward_name='qed', alpha=0.6, beta=0.3, gamma=0.1):
    """
    Calculate rewards for trajectories and create preference pairs
    """
    if not trajectories:
        return [], []
    
    # Extract SMILES strings
    smiles_list = [t["smiles"] for t in trajectories]
    
    # Calculate rewards
    rewards = calculate_rewards(smiles_list, reward_name)
    
    # Calculate diversity
    diversity = calculate_diversity(smiles_list)
    
    # Assign rewards to trajectories
    for i, trajectory in enumerate(trajectories):
        trajectory["reward"] = rewards[i]
    
    # Sort trajectories by reward
    sorted_trajectories = sorted(trajectories, key=lambda x: x["reward"], reverse=True)
    
    # Create preference pairs
    preferred_texts = []
    rejected_texts = []
    
    # Use the top half as preferred and bottom half as rejected
    mid_point = len(sorted_trajectories) // 2
    
    if mid_point > 0:
        for i in range(mid_point):
            preferred = sorted_trajectories[i]
            rejected = sorted_trajectories[-(i+1)]
            
            preferred_texts.append(preferred["full_text"])
            rejected_texts.append(rejected["full_text"])
    
    return preferred_texts, rejected_texts

def train_with_dpo(model, tokenizer, preferred_texts, rejected_texts, output_dir, batch_size=4, learning_rate=5e-5, num_epochs=3):
    """
    Train the model using DPO
    """
    from transformers import DPOTrainer, TrainingArguments

    # Create dataset
    dataset = MoleculeGenerationDataset(tokenizer, preferred_texts, rejected_texts)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        remove_unused_columns=False,
    )
    
    # Create LoRA configuration for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    # Prepare model with LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Set up DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,  # We'll use the same model as reference
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,  # DPO specific parameter
    )
    
    # Train the model
    dpo_trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    return model

def main():
    parser = argparse.ArgumentParser(description='RL training with reward calculation and DPO')
    parser.add_argument('--pool', type=str, required=True, help='Path to candidate pool pickle file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model or model identifier')
    parser.add_argument('--hf_token', type=str, help='HuggingFace token for accessing gated models')
    parser.add_argument('--vllm_endpoint', type=str, help='VLLM API endpoint (e.g., http://0.0.0.0:8000/generate)')
    parser.add_argument('--output_dir', type=str, default='rl_output', help='Directory to save RL results and models')
    parser.add_argument('--num_iterations', type=int, default=3, help='Number of RL iterations')
    parser.add_argument('--samples_per_iter', type=int, default=50, help='Number of samples per iteration')
    parser.add_argument('--positive_examples', type=int, default=10, help='Number of positive examples to use')
    parser.add_argument('--negative_examples', type=int, default=10, help='Number of negative examples to use')
    parser.add_argument('--reward', type=str, default='qed', help='Reward function to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model and tokenizer
    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.hf_token)
    
    # Only load the model if we're not using an endpoint
    model = None
    if not args.vllm_endpoint:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            token=args.hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Load candidate pool
    print(f"Loading candidate pool from {args.pool}...")
    candidate_pool = load_candidate_pool(args.pool)
    print(f"Loaded {len(candidate_pool)} molecules")
    
    # Calculate initial rewards
    print(f"Calculating {args.reward} rewards for molecules...")
    rewards = calculate_rewards(candidate_pool, args.reward)
    
    # Track best molecules and performance
    all_preferred_texts = []
    all_rejected_texts = []
    
    # Database of SMILES to store already tried ones
    generated_smiles_db = set()
    
    # Run RL iterations
    for iteration in range(args.num_iterations):
        print(f"\n{'='*50}")
        print(f"RL Iteration {iteration+1}/{args.num_iterations}")
        print(f"{'='*50}")
        
        # Create (SMILES, reward) pairs and sort by reward
        candidates_with_rewards = list(zip(candidate_pool, rewards))
        sorted_candidates = sorted(candidates_with_rewards, key=lambda x: x[1], reverse=True)
        
        # Find the highest possible reward in the entire pool
        highest_reward = sorted_candidates[0][1]
        print(f"Highest reward in pool: {highest_reward:.4f}")
        
        # Select positive and negative examples
        positive_samples = sorted_candidates[:args.positive_examples]
        negative_samples = sorted_candidates[-args.negative_examples:]
        
        print("\nPositive samples (examples):")
        for i, (smi, reward) in enumerate(positive_samples[:3]):  # Show just a few examples
            print(f"{i+1}. SMILES: {smi}, Reward: {reward:.4f}")
        
        print("\nNegative samples (examples):")
        for i, (smi, reward) in enumerate(negative_samples[:3]):  # Show just a few examples
            print(f"{i+1}. SMILES: {smi}, Reward: {reward:.4f}")
        
        # Generate different types of prompts for exploration
        request_types = ["optimize_qed", "optimize_diversity", "optimize_synthetic", None]
        
        iter_preferred_texts = []
        iter_rejected_texts = []
        
        new_molecules = []
        
        # Try simpler direct requests for SMILES strings
        for i in range(20):  # Increased from 10 to 20 attempts
            # Get top 10 molecules for examples
            top_examples = sorted_candidates[:10]
            
            # Randomly select a high-reward molecule as example
            example_idx = random.randint(0, len(top_examples)-1)
            example_smiles, example_reward = top_examples[example_idx]
            
            # Simple prompt
            simple_prompt = f"""<s>[INST]
I have a molecule with a high drug-likeness score (QED = {example_reward:.4f}):
{example_smiles}

Please generate a new SMILES string for a molecule that might have an even higher QED score.
Focus on creating a valid, drug-like molecule with:
- Good solubility
- Appropriate molecular weight (< 500)
- Reasonable number of hydrogen bond donors/acceptors
- Balance of hydrophilic and lipophilic properties

Only respond with the SMILES string.
[/INST]
"""
            
            print(f"\nTrying simple molecule generation (attempt {i+1}/20)...")
            
            # Try generation
            if args.vllm_endpoint:
                # API based generation
                try:
                    payload = {
                        "prompt": simple_prompt,
                        "temperature": 0.8,
                        "max_tokens": 256
                    }
                    response = requests.post(args.vllm_endpoint, json=payload)
                    response.raise_for_status()
                    
                    response_json = response.json()
                    if "generated_text" in response_json:
                        generated_text = response_json["generated_text"]
                        print(f"Raw output: {generated_text}")
                        
                        # Look for valid SMILES
                        smiles = extract_smiles(generated_text)
                        if smiles:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None:
                                canonical_smiles = Chem.MolToSmiles(mol)
                                if canonical_smiles not in generated_smiles_db:
                                    generated_smiles_db.add(canonical_smiles)
                                    new_molecules.append(canonical_smiles)
                                    print(f"Found valid molecule: {canonical_smiles}")
                except Exception as e:
                    print(f"Error in simple generation: {e}")
            else:
                # Direct model generation
                try:
                    inputs = tokenizer(simple_prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.8,
                            num_return_sequences=1
                        )
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Raw output: {generated_text}")
                    
                    # Look for valid SMILES
                    smiles = extract_smiles(generated_text)
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            canonical_smiles = Chem.MolToSmiles(mol)
                            if canonical_smiles not in generated_smiles_db:
                                generated_smiles_db.add(canonical_smiles)
                                new_molecules.append(canonical_smiles)
                                print(f"Found valid molecule: {canonical_smiles}")
                except Exception as e:
                    print(f"Error in simple generation: {e}")
        
        # Also try the more advanced prompting if needed
        if len(new_molecules) < args.samples_per_iter:
            # For each request type, generate samples
            for req_type in request_types:
                prompt = generate_prompt(positive_samples, negative_samples, req_type)
                
                print(f"\nGenerating trajectories for request type: {req_type or 'default'}...")
                trajectories = sample_trajectories(
                    model, 
                    tokenizer, 
                    prompt,
                    args.vllm_endpoint,
                    num_samples=args.samples_per_iter // len(request_types)
                )
                
                print(f"Generated {len(trajectories)} valid trajectories")
                
                # Create preference pairs
                preferred, rejected = create_preference_pairs(trajectories, args.reward)
                
                iter_preferred_texts.extend(preferred)
                iter_rejected_texts.extend(rejected)
                
                # Extract molecules from trajectories
                for t in trajectories:
                    if t["smiles"] not in generated_smiles_db:
                        generated_smiles_db.add(t["smiles"])
                        new_molecules.append(t["smiles"])
        
        # Calculate rewards for new molecules
        if new_molecules:
            print(f"Generated {len(new_molecules)} new valid molecules")
            new_rewards = calculate_rewards(new_molecules, args.reward)
            
            # Filter to keep only high-quality molecules (above a threshold)
            # For the first iteration, accept all. For later iterations, be more selective
            threshold = 0.7 if iteration == 0 else 0.8
            filtered_molecules = [(smi, rew) for smi, rew in zip(new_molecules, new_rewards) if rew >= threshold]
            
            if filtered_molecules:
                filtered_molecules.sort(key=lambda x: x[1], reverse=True)
                print(f"Keeping {len(filtered_molecules)} molecules with reward >= {threshold:.1f}")
                
                # Add filtered molecules to candidate pool
                for smi, reward in filtered_molecules:
                    if smi not in candidate_pool:
                        candidate_pool.append(smi)
                        rewards.append(reward)
                        
                # Print examples of filtered molecules with their rewards
                print("\nSample of new high-quality molecules:")
                for i, (smi, reward) in enumerate(filtered_molecules[:5]):
                    print(f"{i+1}. SMILES: {smi}, Reward: {reward:.4f}")
                
                # Update best molecules if we found better ones
                if filtered_molecules[0][1] > highest_reward:
                    print(f"\n🎉 New highest reward: {filtered_molecules[0][1]:.4f}, improvement: {filtered_molecules[0][1] - highest_reward:.4f}")
            else:
                print(f"No molecules with reward >= {threshold:.1f}, not adding to pool")
        else:
            print("No new valid molecules generated in this iteration")
        
        # Add to overall preference pairs
        all_preferred_texts.extend(iter_preferred_texts)
        all_rejected_texts.extend(iter_rejected_texts)
        
        print(f"\nAfter iteration {iteration+1}:")
        print(f"Total molecules in pool: {len(candidate_pool)}")
        print(f"Total preference pairs: {len(all_preferred_texts)}")
        
        # Save current state
        state = {
            "iteration": iteration + 1,
            "pool_size": len(candidate_pool),
            "preference_pairs": len(all_preferred_texts),
            "new_molecules": len(new_molecules),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(args.output_dir, f"state_iteration_{iteration+1}.json"), "w") as f:
            json.dump(state, f, indent=2)
        
        # Save current pool
        with open(os.path.join(args.output_dir, f"pool_iteration_{iteration+1}.pkl"), "wb") as f:
            pickle.dump(candidate_pool, f)
        
        # Also save the new molecules separately
        if new_molecules:
            with open(os.path.join(args.output_dir, f"new_molecules_iteration_{iteration+1}.pkl"), "wb") as f:
                pickle.dump(new_molecules, f)
    
    # Train the final model with all preference pairs
    if all_preferred_texts and all_rejected_texts and model is not None:
        print("\nTraining model with DPO using all preference pairs...")
        model_output_dir = os.path.join(args.output_dir, "trained_model")
        os.makedirs(model_output_dir, exist_ok=True)
        
        final_model = train_with_dpo(
            model,
            tokenizer,
            all_preferred_texts,
            all_rejected_texts,
            model_output_dir
        )
        
        print(f"Model trained and saved to {model_output_dir}")
    else:
        if not all_preferred_texts or not all_rejected_texts:
            print("\nNo preference pairs generated, skipping model training")
        elif model is None:
            print("\nUsing API endpoint, local model not available for training")
    
    print("\nRL training complete!")

if __name__ == "__main__":
    main() 