import os
import subprocess
from typing import List, Optional

import pandas as pd


def make_verl_dataset(smiles_list: List[str], output_file: str) -> None:
    """Create a simple verl dataset from a list of SMILES."""
    data = []
    for i, smi in enumerate(smiles_list):
        data.append(
            {
                "data_source": "molleo_pool",
                "prompt": [{"role": "user", "content": f"Improve this molecule: {smi}"}],
                "ability": "chem",
                "reward_model": {"style": "custom"},
                "extra_info": {"index": i},
            }
        )
    pd.DataFrame(data).to_parquet(output_file)


def run_verl_ppo(
    model_path: str,
    train_file: str,
    val_file: str,
    reward_fn: str,
    output_dir: str,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Invoke the verl PPO trainer via subprocess."""
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"actor_rollout_ref.model.path={model_path}",
        f"critic.model.path={model_path}",
        f"reward_model.custom_reward_function.path={reward_fn}",
        "trainer.logger=['console']",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.project_name=molleo_verl",
        f"trainer.experiment_name=run",
        f"trainer.output_dir={output_dir}",
    ]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)
