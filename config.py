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
    
    # Version-based configuration
    VERSION = os.getenv("VERSION")  
    HF_USERNAME = os.getenv("HF_USERNAME") 
  
    # HuggingFace
    HF_TOKEN = os.getenv("HF_TOKEN")
    HUGGINGFACE_REPO_ID = f"{HF_USERNAME}/symptoms_ner_{VERSION}"
    HUGGINGFACE_REPO_ID_BIOBERT = f"{HF_USERNAME}/symptoms_ner_{VERSION}_biobert"
    HUGGINGFACE_MODEL_REPO_ID = f"{HF_USERNAME}/symptom-ner-bert-models" #f"{HF_USERNAME}/symptoms_ner_{VERSION}_models"
    
    # WANDB
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
    SAVE_TO_GCS = os.getenv("SAVE_TO_GCS", "true")
    BUCKET_NAME = os.getenv("BUCKET_NAME", "ner_training_data_results")

settings = Settings()