import re
from rdkit import Chem
from tdc import Oracle


def extract_smiles(text: str) -> str | None:
    """Extract a SMILES string from generated text."""
    match = re.search(r"SMILES:\s*([^\s\n]+)", text)
    if not match:
        match = re.search(r"([A-Za-z0-9@\[\]\(\)=#\+-]+)", text)
    if match:
        smi = match.group(1).strip()
        if Chem.MolFromSmiles(smi) is not None:
            return smi
    return None


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, oracle_name="qed"):
    """Compute reward for verl using a TDC oracle."""
    oracle = Oracle(name=oracle_name)
    smiles = extract_smiles(solution_str)
    if not smiles:
        return 0.0
    try:
        return float(oracle(smiles))
    except Exception:
        return 0.0
