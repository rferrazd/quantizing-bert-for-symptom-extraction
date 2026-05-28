"""
export_to_onnx.py

Exports the V05 BioBERT token-classification checkpoint to ONNX format (fp32).

ONNX is a portable model format. PyTorch is the framework we trained in, but
PyTorch models can only run inside Python with PyTorch installed. ONNX is a
file format that many runtimes can read — including ONNX Runtime Mobile, which
is what the React Native app will use. So the first step in the mobile pipeline
is: take the trained PyTorch weights and write them into an .onnx file.

This script does that for the fp32 (full-precision) version of the model.
Quantization to INT8 is a separate step (see quantize_onnx.py).
"""

from pathlib import Path
import shutil

from optimum.exporters.onnx import main_export
from config import settings

# Where the V05 PyTorch checkpoint lives (input to this script).
SOURCE_MODEL_DIR = Path(
    f"{settings.VERSION}/downloaded_models/dmis-lab/biobert-base-cased-v1.1/run_0"
)

# Where the ONNX artifacts will be written (these are large binary files).
ARTIFACTS_DIR = Path("mobile_app/artifacts/v05")


def export_v05_to_onnx() -> None:
    """
    Export the V05 BioBERT checkpoint to ONNX format (fp32) and copy the
    tokenizer / config / label mapping files alongside it.

    Output files:
        mobile_app/artifacts/v05/model.onnx          # the fp32 ONNX model
        mobile_app/artifacts/v05/tokenizer.json      # tokenizer (copied)
        mobile_app/artifacts/v05/tokenizer_config.json
        mobile_app/artifacts/v05/config.json         # contains id2label / label2id
    """

    # Make sure the output folder exists. parents=True tells mkdir to create any missing parent paths instead of failing.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    
    # 1) Run the actual ONNX export
    # `optimum` is HuggingFace's library for model optimization. Its `main_export`
    # function knows how to take a HuggingFace-style checkpoint (PyTorch weights +
    # config.json) and produce an ONNX graph.
    #
    # `task="token-classification"` tells optimum which model "head" we're using —
    main_export(
        model_name_or_path=str(SOURCE_MODEL_DIR),
        output=str(ARTIFACTS_DIR),
        task="token-classification",
        # opset is the ONNX "version". 14 is broadly compatible with ONNX Runtime
        # Mobile and supports everything BERT needs. Higher numbers add features
        # we don't need; lower numbers may be missing ops.
        opset=14,
    )
    # `main_export` writes the file as `model.onnx` by default and ALSO copies
    # the tokenizer + config files into the output directory

    # 2) Belt-and-braces: explicitly copy the tokenizer files in case optimum
    #    didn't copy something we need. shutil.copy2 preserves metadata.


    for fname in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
        src = SOURCE_MODEL_DIR / fname
        dst = ARTIFACTS_DIR / fname
        if src.exists() and not dst.exists():
            print(f'File {src} was not properly exported to .onnx falling back to shutil.copy2(src, dst)')
            shutil.copy2(src, dst)

    print(f"Wrote ONNX model + tokenizer files to: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    # How to run and verify 
    # From the project root:
    # python -m mobile_app.model_prep.export_to_onnx
    # Verify:
    # ls -lh mobile_app/artifacts/v05/ (-lh means to print in human readable size not in raw byte counts)
    export_v05_to_onnx()