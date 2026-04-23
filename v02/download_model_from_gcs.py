"""
Download the trained NER model from Google Cloud Storage.

Credentials required:
  - A GCP service account key or Application Default Credentials (ADC) with at least:
      storage.objects.list  and  storage.objects.get  on the target bucket.
  - The calling identity must also have serviceusage.services.use on the project
    (i.e. the Cloud Storage API must be enabled and the IAM role must grant it).
  - Quickest fix: run `gcloud auth application-default login` locally, or set
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

If you receive a 403 error, fall back to gsutil:
  gsutil -m cp -r gs://<BUCKET>/v02/runs/dmis-lab/biobert-base-cased-v1.1/run_2 \
      <repo_root>/v02/downloaded_models/dmis-lab/biobert-base-cased-v1.1/run_2
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root from this file's location (works regardless of cwd)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent  # v02/ -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings          # noqa: E402
from gcp_utils import download_from_gcs  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults (mirror the notebook exactly)
# ---------------------------------------------------------------------------
DEFAULT_BUCKET = settings.BUCKET_NAME                              # "ner_training_data_results"
DEFAULT_GCS_PREFIX = "v02/runs/dmis-lab/biobert-base-cased-v1.1/run_2"
DEFAULT_LOCAL_DIR = str(
    REPO_ROOT / "v02" / "downloaded_models" / "dmis-lab" / "biobert-base-cased-v1.1" / "run_2"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the BioBERT NER model from GCS to a local directory."
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"GCS bucket name (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_GCS_PREFIX,
        help=f"GCS object prefix / path inside the bucket (default: {DEFAULT_GCS_PREFIX})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Local destination directory (default: {DEFAULT_LOCAL_DIR})",
    )
    args = parser.parse_args()

    local_dir = Path(args.output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source : gs://{args.bucket}/{args.prefix}")
    print(f"Dest   : {local_dir}")
    print()

    result = download_from_gcs(
        gcs_path=args.prefix,
        local_path=str(local_dir),
        bucket_name=args.bucket,
    )

    if result:
        print(f"\nSuccess — model files are at:\n  {local_dir}")
        files = sorted(local_dir.rglob("*"))
        print(f"  ({len([f for f in files if f.is_file()])} file(s) downloaded)")
    else:
        print("\nDownload failed. See error above for details.", file=sys.stderr)
        print("Fix credentials, then retry, or use gsutil:", file=sys.stderr)
        print(
            f'  gsutil -m cp -r "gs://{args.bucket}/{args.prefix}" "{local_dir}"',
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
