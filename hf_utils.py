"""Utilities for interacting with Hugging Face Hub"""

from huggingface_hub import create_branch, HfApi


def ensure_branch_exists(repo_id: str, branch_name: str, repo_type: str = "model") -> bool:
    """
    Ensure a branch exists in a Hugging Face Hub repository.
    Creates the branch if it doesn't exist, otherwise does nothing.
    
    Args:
        repo_id: The repository ID (e.g., "username/repo-name")
        branch_name: The name of the branch to create/check
        repo_type: Type of repository ("model", "dataset", or "space"). Defaults to "model"
    
    Returns:
        True if branch exists or was created successfully, False otherwise
    """
    try:
        api = HfApi()
        # Check if branch exists
        try:
            refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type)
            branch_names = [branch.name for branch in refs.branches] if refs.branches else []
            
            if branch_name not in branch_names:
                print(f"Branch '{branch_name}' doesn't exist. Creating it from 'main'...")
                create_branch(repo_id=repo_id, branch=branch_name, repo_type=repo_type)
                print(f"✓ Branch '{branch_name}' created successfully!")
                return True
            else:
                print(f"✓ Branch '{branch_name}' already exists.")
                return True
        except Exception as e:
            # If checking fails, try creating the branch anyway
            print(f"Could not check existing branches, attempting to create: {e}")
            create_branch(repo_id=repo_id, branch=branch_name, repo_type=repo_type)
            print(f"✓ Branch '{branch_name}' created successfully!")
            return True
    except Exception as e:
        # Branch might already exist, or repo might not exist yet
        error_msg = str(e).lower()
        if "already exists" in error_msg or "409" in error_msg:
            print(f"✓ Branch '{branch_name}' already exists.")
            return True
        else:
            print(f"⚠ Warning: Could not create branch: {e}")
            return False

