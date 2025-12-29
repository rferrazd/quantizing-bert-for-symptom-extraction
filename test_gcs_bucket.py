"""
Test script to verify GCS bucket upload functionality
Tests uploading files and directories to the ner-training-data-results bucket
Uses functions from gcp_utils.py to test directory upload functionality
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from gcp_utils import upload_to_gcs, verify_upload, list_bucket_files

# Configuration
BUCKET_NAME = "ner-training-data-results"
TEST_DIR_NAME = "test_run_directory"
GCS_TEST_PREFIX = "test-uploads"  # Organize test files in a subfolder

def create_test_directory():
    """Create a test directory structure similar to training output"""
    # Remove test directory if it exists
    if os.path.exists(TEST_DIR_NAME):
        shutil.rmtree(TEST_DIR_NAME)
    
    # Create main test directory
    test_dir = Path(TEST_DIR_NAME)
    test_dir.mkdir(exist_ok=True)
    
    # Create files in root of test directory
    # 1. Metrics JSON files
    val_metrics = {
        "f1": 0.85,
        "precision": 0.87,
        "recall": 0.83,
        "accuracy": 0.92,
        "test_timestamp": datetime.now().isoformat()
    }
    with open(test_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    
    test_metrics = {
        "f1": 0.88,
        "precision": 0.90,
        "recall": 0.86,
        "accuracy": 0.94,
        "test_timestamp": datetime.now().isoformat()
    }
    with open(test_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # 2. Create a dummy plot file (just a text file for testing)
    with open(test_dir / "val_f1_bins_plot.png", "w") as f:
        f.write("This is a dummy plot file for testing directory uploads.\n")
    
    with open(test_dir / "test_f1_bins_plot.png", "w") as f:
        f.write("This is a dummy plot file for testing directory uploads.\n")
    
    # 3. Create a checkpoint subdirectory (simulating model checkpoint)
    checkpoint_dir = test_dir / "checkpoint-447"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create files in checkpoint directory
    trainer_state = {
        "best_global_step": 447,
        "best_metric": 0.85,
        "epoch": 1.0,
        "global_step": 447
    }
    with open(checkpoint_dir / "trainer_state.json", "w") as f:
        json.dump(trainer_state, f, indent=2)
    
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({"model_type": "test-model", "num_labels": 100}, f, indent=2)
    
    with open(checkpoint_dir / "tokenizer.json", "w") as f:
        f.write("This is a dummy tokenizer file.\n")
    
    # Create a dummy model file
    with open(checkpoint_dir / "model.safetensors", "w") as f:
        f.write("This is a dummy model weights file for testing.\n")
    
    print(f"✓ Created test directory structure:")
    print(f"  - {TEST_DIR_NAME}/")
    print(f"    - val_metrics.json")
    print(f"    - test_metrics.json")
    print(f"    - val_f1_bins_plot.png")
    print(f"    - test_f1_bins_plot.png")
    print(f"    - checkpoint-447/")
    print(f"      - trainer_state.json")
    print(f"      - config.json")
    print(f"      - tokenizer.json")
    print(f"      - model.safetensors")
    
    return str(test_dir)


def main():
    print("=" * 60)
    print("GCS BUCKET UPLOAD TEST - Directory Upload")
    print("=" * 60)
    
    # Step 1: Create test directory structure
    print("\n[Step 1] Creating test directory structure...")
    test_dir_path = create_test_directory()
    
    # Step 2: Upload entire directory
    print(f"\n[Step 2] Uploading entire directory to gs://{BUCKET_NAME}/...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_dir_path = f"{GCS_TEST_PREFIX}/{timestamp}/{TEST_DIR_NAME}"
    
    success = upload_to_gcs(
        local_path=test_dir_path,
        gcs_path=gcs_dir_path,
        bucket_name=BUCKET_NAME
    )
    
    if not success:
        print("\n✗ Upload failed. Check your GCP credentials and permissions.")
        # Cleanup
        if os.path.exists(TEST_DIR_NAME):
            shutil.rmtree(TEST_DIR_NAME)
        return
    
    # Step 3: Verify uploads (check a few key files)
    print(f"\n[Step 3] Verifying key files were uploaded...")
    verify_upload(BUCKET_NAME, f"{gcs_dir_path}/val_metrics.json")
    verify_upload(BUCKET_NAME, f"{gcs_dir_path}/test_metrics.json")
    verify_upload(BUCKET_NAME, f"{gcs_dir_path}/checkpoint-447/trainer_state.json")
    verify_upload(BUCKET_NAME, f"{gcs_dir_path}/checkpoint-447/model.safetensors")
    
    # Step 4: List all files in uploaded directory
    print(f"\n[Step 4] Listing all files in uploaded directory...")
    list_bucket_files(BUCKET_NAME, prefix=gcs_dir_path)
    
    # Step 5: Test single file upload (for comparison)
    print(f"\n[Step 5] Testing single file upload...")
    single_file_path = f"{TEST_DIR_NAME}/single_test_file.txt"
    with open(single_file_path, "w") as f:
        f.write("This is a single file upload test.\n")
    
    gcs_single_path = f"{GCS_TEST_PREFIX}/{timestamp}/single_test_file.txt"
    upload_to_gcs(single_file_path, gcs_single_path, BUCKET_NAME)
    verify_upload(BUCKET_NAME, gcs_single_path)
    
    # Step 6: Cleanup local test directory
    print(f"\n[Step 6] Cleaning up local test directory...")
    if os.path.exists(TEST_DIR_NAME):
        shutil.rmtree(TEST_DIR_NAME)
        print(f"✓ Removed {TEST_DIR_NAME}")
    if os.path.exists(single_file_path):
        os.remove(single_file_path)
        print(f"✓ Removed {single_file_path}")
    
    print("\n" + "=" * 60)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nYour directory is available at:")
    print(f"  gs://{BUCKET_NAME}/{gcs_dir_path}")
    print(f"\nYou can view it in the GCP Console:")
    print(f"  https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{GCS_TEST_PREFIX}/{timestamp}")
    print(f"\nThe directory structure is preserved in GCS, including the checkpoint subdirectory!")

if __name__ == "__main__":
    main()
