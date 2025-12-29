import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Settings loads environment variables for the project.
    NOTE: For cache/config directories, we set defaults using os.environ.setdefault to ensure 
    directories are defined even if not present in the environment. For secrets and service tokens, 
    we use os.getenv so they remain unset if not specified.
    
    Difference:
      - os.environ.setdefault("KEY", "default") sets a default value for a process-wide environment variable
        (visible to subprocesses and libraries), but only if not already set.
        Use this for configuration values (like cache directories) where a sensible default is needed.
      - os.getenv("KEY") only reads the current environment variable 'KEY', returning None if unset.
        Use this for sensitive keys and values, so you don't hard-code secrets in your codebase.
    
    In this class, secrets/tokens are loaded with getenv (require explicit setting), while any project-wide 
    configuration defaults are set via setdefault (see above, at top of file).
    """

    BIO_PORTAL_API_KEY = os.getenv("BIO_PORTAL_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    HUGGINGFACE_REPO_ID = os.getenv("HUGGINGFACE_REPO_ID")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    USE_WANDB = os.getenv("USE_WANDB")
    WANDB_PROJECT = os.environ.setdefault("WANDB_PROJECT", "symptom-ner")
    # ---------------------------------
    # For GCP training
    # ---------------------------------
    HF_HOME = os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    HF_DATASETS_CACHE = os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/huggingface/datasets")
    HF_TRANSFORMERS_CACHE = os.environ.setdefault("HF_TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")
    HF_HUB_CACHE = os.environ.setdefault("HF_HUB_CACHE", "/tmp/huggingface/hub")
    # GCS Bucket
    BUCKET_NAME = os.getenv("BUCKET_NAME", "ner-training-data-results")

settings = Settings()