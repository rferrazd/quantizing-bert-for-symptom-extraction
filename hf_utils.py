"""
hf_utils.py

Utilities for interacting with the Hugging Face Hub. Covers dataset upload
(upsert_to_hf_repo) and branch management (ensure_branch_exists).
"""
import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from huggingface_hub import create_branch, HfApi


def upsert_to_hf_repo(
    jsonl_path: str | Path,
    repo_id: str,
    split_name: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> None:
    """
    Upload a single JSONL file as a named split to a HuggingFace dataset repo.

    If the repo does not exist it is created automatically (private by default).
    If the split already exists on the Hub it is overwritten. If revision is
    given the data is pushed to that branch; otherwise it lands on main.

    Args:
        jsonl_path:  Path to the wordpiece-aligned JSONL file to upload.
        repo_id:     HuggingFace dataset repo, e.g. "username/symptom-ner-dataset-v04".
        split_name:  Name for this split on the Hub, e.g. "train", "template_ood".
        token:       HF auth token. Falls back to the cached login token if None.
        revision:    Branch name to push to. Pushes to main if None.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    print(f"Loading {jsonl_path.name} ...")
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dataset = Dataset.from_list(rows)
    print(f"  {len(dataset):,} samples | features: {list(dataset.features.keys())}")

    kwargs = dict(
        repo_id=repo_id,
        split=split_name,
        token=token,
        private=True,
    )
    if revision:
        kwargs["revision"] = revision

    print(f"Pushing to {repo_id}  split={split_name}  branch={revision or 'main'} ...")
    dataset.push_to_hub(**kwargs)
    print(f"Done. https://huggingface.co/datasets/{repo_id}/viewer/{split_name}")


def ensure_branch_exists(repo_id: str, branch_name: str, repo_type: str = "model") -> bool:
    """
    Ensure a branch exists in a Hugging Face Hub repository.
    Creates the branch if it doesn't exist, otherwise does nothing.

    Args:
        repo_id:     The repository ID (e.g., "username/repo-name").
        branch_name: The name of the branch to create/check.
        repo_type:   Type of repository ("model", "dataset", or "space").

    Returns:
        True if branch exists or was created successfully, False otherwise.
    """
    try:
        api = HfApi()
        try:
            refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type)
            branch_names = [branch.name for branch in refs.branches] if refs.branches else []

            if branch_name not in branch_names:
                print(f"Branch '{branch_name}' doesn't exist. Creating it from 'main'...")
                create_branch(repo_id=repo_id, branch=branch_name, repo_type=repo_type)
                print(f"Branch '{branch_name}' created successfully.")
                return True
            else:
                print(f"Branch '{branch_name}' already exists.")
                return True
        except Exception as e:
            print(f"Could not check existing branches, attempting to create: {e}")
            create_branch(repo_id=repo_id, branch=branch_name, repo_type=repo_type)
            print(f"Branch '{branch_name}' created successfully.")
            return True
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "409" in error_msg:
            print(f"Branch '{branch_name}' already exists.")
            return True
        else:
            print(f"Warning: Could not create branch: {e}")
            return False
