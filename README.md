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

### 4. Single-Objective Pipeline with Optional DPO

```bash
python single_objective/pipeline.py --n 1000 --iterations 3 --m 10 --reward qed \
    --vllm_endpoint http://localhost:8000/generate --output_dir single_output
```

Add `--dpo` to perform Direct Preference Optimization fine-tuning after the search.

The RL reward combines the improvement in maximum and mean rewards of the newly
generated molecules compared to the positive samples along with their diversity:

```
reward = alpha * (new_reward_max - positive_reward_max)
         + beta * (new_reward_mean - positive_reward_mean)
         + gamma * diversity(new_samples)
```

### 5. Multi-Objective Pipeline with Optional DPO

```bash
python multi_objective/pipeline.py --n 1000 --iterations 3 --m 10 \
    --max_obj jnk3 qed --min_obj sa \
    --vllm_endpoint http://localhost:8000/generate --output_dir multi_output
```

This variant accepts multiple objectives to maximize (`--max_obj`) and minimize
(`--min_obj`). Use `--dpo` to fine-tune the model with preference pairs after the
search completes.

### 6. Compute Metrics for a Run

After running the optimization pipelines, use `compute_metrics.py` to calculate
the evaluation metrics reported in the MolLEO paper. For single-objective runs
provide the reward name:

```bash
python compute_metrics.py --run_dir results/run_XXXX --reward qed
```

For multi-objective runs specify the objectives to maximize and minimize:

```bash
python compute_metrics.py --run_dir multi_output --max_obj jnk3 qed --min_obj sa
```

The script outputs the top‑10 AUC and final statistics for single-objective
runs or the summed AUC, hypervolume and diversity metrics for multi-objective
runs. Use `--output metrics.json` to save the values to a file.

### 7. RLVR Training

`rlvr_train.py` implements reinforcement learning with variance reduction. It
attaches a small value head to the language model and updates the model using
an actor–critic objective. Rewards are computed using the same formula as the
single-objective pipeline.

```bash
python rlvr_train.py --pool candidate_pool.pkl --model Qwen/Qwen2-7B-Instruct \
    --vllm_endpoint http://localhost:8000/generate
```

The script saves the fine‑tuned model and value head to the `rlvr_model/`
directory.

### 8. GRPO Training (RLVR + DPO)

`grpo_train.py` runs an RLVR loop and then fine‑tunes the model with Direct
Preference Optimization. This produces a policy optimized with variance
reduction and preference learning.

```bash
python grpo_train.py --pool candidate_pool.pkl --model Qwen/Qwen2-7B-Instruct \
    --vllm_endpoint http://localhost:8000/generate
```

The resulting model is stored in `grpo_model/`.

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

## 9. Training with verl

Experimental support for the [verl](https://github.com/volcengine/verl) library
is provided via the `verl_train.py` script. After installing `verl` you can fine
tune a model using PPO:

```bash
pip install verl
python verl_train.py --model Qwen/Qwen2.5-0.5B-Instruct --n 1000 --output_dir verl_output
```

This converts a candidate pool into a simple verl dataset and launches
`verl.trainer.main_ppo` with the custom reward defined in `verl_reward.py`.

## License

[License information]

## Acknowledgements

This work builds on the MolLEO framework and utilizes open-source LLMs hosted on VLLM for molecule generation.
