"""
create_hf_model_repo.py

Creates the HuggingFace model repo and all per-run branches for V05 training
before deploy_to_gcp.sh is run. Branch names mirror the exact logic in
v05/7_trainer_gcp.py so the trainer can push without hitting a 404.

Run: python v05/create_hf_model_repo.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import create_repo, HfApi
from config import settings
from hyperparam_sets import biobert_hyperparams_v05


def branch_name(idx: int, hp: dict) -> str:
    """
    Reproduces the branch-name logic from v05/7_trainer_gcp.py so the names
    created here match exactly what the trainer will try to push to.
    """
    model_short = hp["model_name"].split("/")[-1].replace("-", "_")
    branch = (
        f"run-{idx}-{model_short}"
        f"-lr{hp['lr']}"
        f"-bs{hp['batch_size']}"
        f"-ep{hp['epoch']}"
    )
    return branch.replace(".", "_").replace("e-", "e").replace("e+", "e")


def main() -> None:
    """Creates the V05 model repo and one branch per hyperparameter set."""
    repo_id = settings.HUGGINGFACE_MODEL_REPO_ID
    token = settings.HF_TOKEN
    api = HfApi()

    # Create the repo (no-op if it already exists)
    print(f"Creating repo: {repo_id} ...")
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    print(f"  Done.")

    # Create one branch per hyperparam run
    for idx, hp in enumerate(biobert_hyperparams_v05):
        b = branch_name(idx, hp)
        print(f"Creating branch: {b} ...")
        try:
            api.create_branch(repo_id=repo_id, branch=b, repo_type="model", token=token)
            print(f"  Created.")
        except Exception as e:
            if "already exists" in str(e).lower() or "reference already exists" in str(e).lower():
                print(f"  Already exists — skipping.")
            else:
                raise

    print(f"\nAll done. https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
