# Molecular RL with Large Language Models

This repository contains a set of scripts for using large language models (LLMs) to generate and optimize molecules through reinforcement learning.

## Overview

The process works in three main steps:

1. **Initialize a candidate pool**: Randomly sample molecules from ZINC database
2. **Run iterations**: For each iteration:
   - Sample molecules from the pool
   - Score them with reward oracles
   - Prompt LLM to generate improved molecules based on examples
   - Calculate rewards and update the pool
3. **Track performance**: Record and plot performance metrics across iterations

## Scripts

- `initialize_pool.py`: Create an initial pool of molecules from ZINC database
- `run_iteration.py`: Run a single iteration of the RL algorithm
- `run_multi_iterations.py`: Run multiple iterations and track performance
- `vllm_server.py`: Helper script to start a VLLM server with Qwen or other models

## Requirements

- Python 3.7+
- RDKit
- TDC (Therapeutics Data Commons)
- Qwen2-7B-Instruct model hosted on VLLM
- NumPy, Matplotlib
- Pickle
- Requests

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd <repository_directory>

# Install the dependencies
pip install rdkit tdc numpy matplotlib requests
```

## VLLM Setup for Qwen Models

To set up VLLM with Qwen2-7B-Instruct:

1. **Get a HuggingFace Token**:
   - Create an account on [HuggingFace](https://huggingface.co)
   - Visit [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and accept the terms of use
   - Generate an access token at https://huggingface.co/settings/tokens

2. **Start the VLLM server**:
   ```bash
   python vllm_server.py --hf-token YOUR_HUGGINGFACE_TOKEN --trust-remote-code
   ```

3. **Alternative models**:
   If you're having issues with Qwen models, you can use other models:
   ```bash
   # List available models
   python vllm_server.py --list-models
   
   # Use Mistral model instead
   python vllm_server.py --model mistralai/Mistral-7B-Instruct-v0.2 --trust-remote-code
   ```

Your VLLM endpoint will be `http://localhost:8000/generate` or replace with your server's address.

## Usage

### 1. Initialize a Candidate Pool

```bash
python initialize_pool.py --n 1000 --seed 42 --output candidate_pool.pkl
```

Options:
- `--n`: Number of molecules to sample (default: 1000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Output file to save candidate pool (default: candidate_pool.pkl)

### 2. Run a Single Iteration

```bash
python run_iteration.py --pool candidate_pool.pkl --m 10 --reward qed --vllm_endpoint http://localhost:8000/generate
```

Options:
- `--pool`: Path to candidate pool pickle file
- `--m`: Number of positive/negative samples (default: 10)
- `--reward`: Reward function to use (default: qed)
- `--vllm_endpoint`: VLLM API endpoint for the model
- `--output`: Output file for updated pool (default: auto-generated)
- `--sample_method`: Method for sampling from the pool (uniform or exp, default: exp)

### 3. Run Multiple Iterations

```bash
python run_multi_iterations.py --initial_pool candidate_pool.pkl --iterations 5 --m 10 --reward qed --vllm_endpoint http://localhost:8000/generate
```

Options:
- `--initial_pool`: Path to initial candidate pool pickle file
- `--iterations`: Number of iterations to run (default: 5)
- `--m`: Number of positive/negative samples per iteration (default: 10)
- `--reward`: Reward function to use (default: qed)
- `--vllm_endpoint`: VLLM API endpoint for the model
- `--output_dir`: Directory to save results (default: results)
- `--sample_method`: Method for sampling from the pool (uniform or exp, default: exp)

## Reward Functions

Available reward functions from TDC Oracle:
- `qed`: Quantitative Estimate of Drug-likeness
- `sa`: Synthetic Accessibility
- `jnk3`: JNK3 inhibition
- `gsk3b`: GSK3Beta inhibition
- `drd2`: DRD2 inhibition

## Example Workflow

```bash
# Step 1: Initialize a pool of 1000 molecules
python initialize_pool.py --n 1000 --output candidate_pool.pkl

# Step 2: Start the VLLM server with your HuggingFace token
python vllm_server.py --hf-token YOUR_HUGGINGFACE_TOKEN --trust-remote-code

# Step 3: Run 5 iterations of the RL algorithm
python run_multi_iterations.py --initial_pool candidate_pool.pkl --iterations 5 --reward qed --vllm_endpoint http://localhost:8000/generate --output_dir results
```

## Troubleshooting

If you encounter issues with the Qwen model:

1. **Authentication errors**: Make sure you've accepted the model's terms of use on HuggingFace and provided a valid token
2. **Model not found**: Check the correct model name using `--list-models` flag
3. **Memory issues**: Reduce `--gpu-memory-utilization` to 0.7 or lower
4. **Alternative models**: Try other models like Mistral-7B-Instruct-v0.2 which may have fewer access restrictions

## Results

The results will be saved in the specified output directory:
- Pickle files for each iteration's candidate pool
- JSON file with rewards for each iteration
- Plot of rewards over iterations
- Symlink to the best performing pool

## License

[License information]

## Acknowledgements

This work builds on the MolLEO framework and utilizes open-source LLMs hosted on VLLM for molecule generation.
