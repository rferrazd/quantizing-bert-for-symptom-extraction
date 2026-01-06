""" 
File with code for GCP services
"""

import os
import json
from pathlib import Path
from google.cloud import storage
from datetime import datetime

# ============ 
# GCS 
# ============

def upload_to_gcs(local_path: str, gcs_path: str, bucket_name: str):
    """
    Upload a file or directory to GCS bucket.
    If local_path is a directory, uploads all files recursively preserving directory structure.
    
    Args:
        local_path: Local file or directory path
        gcs_path: Destination path in GCS (e.g., "test-uploads/file.txt" or "runs/model/run_0")
        bucket_name: GCS bucket name
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        local_path_obj = Path(local_path)
        
        if local_path_obj.is_file():
            # Upload single file
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded: {local_path} → gs://{bucket_name}/{gcs_path}")
            return True
        elif local_path_obj.is_dir():
            # Upload directory recursively
            uploaded_count = 0
            for file_path in local_path_obj.rglob("*"):
                if file_path.is_file():
                    # Get relative path from the directory being uploaded
                    relative_path = file_path.relative_to(local_path_obj)
                    # Construct GCS path: if gcs_path ends with /, append relative_path, otherwise join them
                    if gcs_path.endswith("/"):
                        gcs_file_path = f"{gcs_path}{relative_path}".replace("\\", "/")
                    else:
                        gcs_file_path = f"{gcs_path}/{relative_path}".replace("\\", "/")
                    
                    blob = bucket.blob(gcs_file_path)
                    blob.upload_from_filename(str(file_path))
                    uploaded_count += 1
                    print(f"✅ Uploaded: {file_path} → gs://{bucket_name}/{gcs_file_path}")
            
            print(f"✓ Uploaded directory: {local_path} ({uploaded_count} files) → gs://{bucket_name}/{gcs_path}")
            return True
        else:
            print(f"❌ Path does not exist: {local_path}")
            return False
    except Exception as e:
        print(f"❌ Failed to upload {local_path}: {e}")
        return False
    
def verify_upload(bucket_name: str, gcs_path: str):
    """
    Verify that a file or directory exists in GCS.
    For directories, checks if any files exist with the given prefix.
    For files, checks if the specific blob exists.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Normalize the path: ensure it ends with / for directories, or check if it's a file
        # If gcs_path doesn't end with /, treat it as potentially a directory prefix
        # First, try checking as a single file blob
        blob = bucket.blob(gcs_path)
        
        if blob.exists():
            # It's a file
            print(f"✓ Verified file: gs://{bucket_name}/{gcs_path} exists")
            print(f"  Size: {blob.size} bytes")
            print(f"  Created: {blob.time_created}")
            return True
        else:
            # Check if it's a directory (prefix) - look for any files with this prefix
            # Ensure prefix ends with / for proper directory matching
            prefix = gcs_path if gcs_path.endswith("/") else f"{gcs_path}/"
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
            
            if blobs:
                # Directory exists (has at least one file)
                # Count all files in the directory
                all_blobs = list(bucket.list_blobs(prefix=prefix))
                total_size = sum(blob.size for blob in all_blobs)
                print(f"✓ Verified directory: gs://{bucket_name}/{gcs_path}/ exists")
                print(f"  Files found: {len(all_blobs)}")
                print(f"  Total size: {total_size:,} bytes")
                return True
            else:
                print(f"❌ Path not found: gs://{bucket_name}/{gcs_path}")
                return False
    except Exception as e:
        print(f"❌ Failed to verify upload: {e}")
        return False

def list_bucket_files(bucket_name: str, prefix: str = None):
    """
    List files in the bucket (optionally filtered by prefix)
    
    Args:
        bucket_name: Name of the GCS bucket.
        prefix: (Optional) A prefix string to filter the listed files. 
            In GCS, a "prefix" acts like a folder filter: for example,
            if your bucket contains objects with names like 
              - "test-uploads/20240401/fileA.txt"
              - "test-uploads/20240402/fileB.json"
              - "other-dir/img.png"
            Passing prefix="test-uploads/20240401" will list only items
            whose names start with "test-uploads/20240401".
            Note: GCS does not have real folders; all files are flat, but slashes (/) in names
            are used to simulate directory hierarchies. The prefix parameter operates on these names.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        print(f"\n📁 Files in gs://{bucket_name}/" + (f"{prefix}/" if prefix else ""))
        blobs = bucket.list_blobs(prefix=prefix) if prefix else bucket.list_blobs()
        
        file_count = 0
        for blob in blobs:
            print(f"  - {blob.name} ({blob.size} bytes)")
            file_count += 1
        
        if file_count == 0:
            print("  (empty)")
        else:
            print(f"\n  Total: {file_count} file(s)")
        
        return True
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return False