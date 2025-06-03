#!/usr/bin/env python
"""Compute evaluation metrics for MolLEO runs.

This script reads the pool files produced by ``single_objective/pipeline.py`` or
``multi_objective/pipeline.py`` and calculates the metrics reported in the
MolLEO paper. For single-objective runs we compute the top-10 AUC of the
optimized property. For multi-objective runs we additionally compute the
hypervolume of the Pareto set as well as structural and objective diversity.
"""

import argparse
import glob
import os
import pickle
import re
from typing import List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tdc import Oracle

try:
    from deap.tools._hypervolume import hv
except Exception:  # pragma: no cover - optional dependency
    hv = None


def load_pools(run_dir: str) -> List[List[str]]:
    """Load candidate pools sorted by iteration."""
    files = sorted(glob.glob(os.path.join(run_dir, "pool_iteration_*.pkl")), key=lambda x: int(re.findall(r"(\d+)", x)[-1]))
    pools = []
    for fp in files:
        with open(fp, "rb") as f:
            pools.append(pickle.load(f))
    return pools


def calculate_rewards(smiles_list: List[str], reward_name: str) -> List[float]:
    oracle = Oracle(name=reward_name)
    scores = []
    for smi in smiles_list:
        try:
            scores.append(oracle(smi))
        except Exception:
            scores.append(0.0)
    return scores


def calculate_diversity(smiles_list: List[str]) -> float:
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    if len(mols) < 2:
        return 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - float(np.mean(sims))


def topk_auc(values: List[float]) -> float:
    """Simple trapezoidal AUC with unit spacing."""
    if len(values) < 2:
        return 0.0
    return float(np.trapz(values, dx=1))


def compute_single_metrics(pools: List[List[str]], reward: str, k: int = 10):
    top_means = []
    for pool in pools:
        scores = calculate_rewards(pool, reward)
        top_means.append(float(np.mean(sorted(scores, reverse=True)[:k])))
    auc = topk_auc(top_means)
    final_scores = calculate_rewards(pools[-1], reward)
    return {
        "top10_auc": auc,
        "final_top10_mean": float(np.mean(sorted(final_scores, reverse=True)[:k])),
        "final_max": float(max(final_scores)),
        "final_mean": float(np.mean(final_scores)),
        "final_diversity": calculate_diversity(pools[-1]),
    }


def calculate_multi_rewards(smiles: List[str], max_obj: List[str], min_obj: List[str]) -> List[List[float]]:
    max_oracles = [Oracle(name=o) for o in max_obj]
    min_oracles = [Oracle(name=o) for o in min_obj]
    res = []
    for s in smiles:
        vals = []
        for o in max_oracles:
            try:
                vals.append(o(s))
            except Exception:
                vals.append(0.0)
        for o in min_oracles:
            try:
                val = o(s)
                if o.name.lower() == "sa":
                    vals.append(1 - ((val - 1) / 9))
                else:
                    vals.append(1 - val)
            except Exception:
                vals.append(0.0)
        res.append(vals)
    return res


def pareto_front(points: List[List[float]]) -> List[List[float]]:
    front = []
    for p in points:
        dominated = False
        for q in points:
            if all(qi >= pi for qi, pi in zip(q, p)) and any(qi > pi for qi, pi in zip(q, p)):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def compute_multi_metrics(pools: List[List[str]], max_obj: List[str], min_obj: List[str], k: int = 10):
    per_iter = []
    for pool in pools:
        scores = calculate_multi_rewards(pool, max_obj, min_obj)
        sums = [sum(v) for v in scores]
        per_iter.append(float(np.mean(sorted(sums, reverse=True)[:k])))
    auc_sum = topk_auc(per_iter)

    final_scores = calculate_multi_rewards(pools[-1], max_obj, min_obj)
    final_sums = [sum(v) for v in final_scores]
    final_top10 = float(np.mean(sorted(final_sums, reverse=True)[:k]))

    results = {
        "top10_auc_sum": auc_sum,
        "final_top10_sum": final_top10,
        "final_structural_diversity": calculate_diversity(pools[-1]),
    }

    # Objective diversity
    if len(final_scores) > 1:
        dists = []
        for i in range(len(final_scores)):
            for j in range(i + 1, len(final_scores)):
                dists.append(np.linalg.norm(np.array(final_scores[i]) - np.array(final_scores[j])))
        results["final_objective_diversity"] = float(np.mean(dists))
    else:
        results["final_objective_diversity"] = 0.0

    # Hypervolume
    if hv is not None and final_scores:
        ref = [0.0] * len(final_scores[0])
        hv_val = hv.hypervolume(-np.array(final_scores), -np.array(ref))
        results["hypervolume"] = float(hv_val)
    else:
        results["hypervolume"] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute MolLEO evaluation metrics")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing pool_iteration_*.pkl")
    parser.add_argument("--reward", type=str, default=None, help="Reward name for single-objective runs")
    parser.add_argument("--max_obj", nargs="*", default=None, help="Objectives to maximize for multi-objective runs")
    parser.add_argument("--min_obj", nargs="*", default=None, help="Objectives to minimize for multi-objective runs")
    parser.add_argument("--k", type=int, default=10, help="Top-k value (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to store metrics")
    args = parser.parse_args()

    pools = load_pools(args.run_dir)
    if not pools:
        raise ValueError("No pool files found in run_dir")

    if args.reward:
        metrics = compute_single_metrics(pools, args.reward, args.k)
    else:
        if not args.max_obj:
            raise ValueError("--max_obj required for multi-objective metrics")
        metrics = compute_multi_metrics(pools, args.max_obj, args.min_obj or [], args.k)

    if args.output:
        with open(args.output, "w") as f:
            import json
            json.dump(metrics, f, indent=2)
    else:
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
