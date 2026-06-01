"""
bundle_app_assets.py

Copies a model version's artifacts into the React Native app's asset folder
under generic, version-less filenames so the app code never references a
specific version. Switching the app to a new model version (e.g. V05 -> V06)
is therefore a single re-run of this script against the new artifacts dir.

Outputs (always the same names, regardless of source version):
    assets/model/model.onnx   <- chosen .onnx (default: model_int8.onnx)
    assets/model/vocab.json    <- vocab.txt converted to a JSON array (index = id)
    assets/model/config.json   <- copied verbatim (provides id2label, etc.)

Usage:
    python bundle_app_assets.py --version v05 --quant int8
    python bundle_app_assets.py --artifacts-dir /path/to/v06 --quant fp32
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def resolve_artifacts_dir(version: str | None, artifacts_dir: str | None) -> Path:
    """
    Determines the source artifacts directory from either an explicit path or a
    version name (resolved relative to this script's mobile_app/artifacts/<version>).
    Returns the resolved Path and raises if it does not exist.
    """
    if artifacts_dir is not None:
        src = Path(artifacts_dir).expanduser().resolve()
    else:
        mobile_app_root = Path(__file__).resolve().parents[1]
        src = mobile_app_root / "artifacts" / version
    if not src.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {src}")
    return src


def _atomic_write_bytes(src: Path, dest: Path) -> None:
    """
    Copies `src` to `dest` atomically: writes to `dest.tmp` then renames. Prevents
    a half-copied model.onnx (interrupted run, low disk) from being loaded later
    and surfacing as a cryptic protobuf error inside ONNX Runtime.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _atomic_write_text(text: str, dest: Path) -> None:
    """Writes text to `dest` via a .tmp + rename, same rationale as the bytes variant."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, dest)


def vocab_txt_to_json(vocab_txt_path: Path, out_path: Path) -> int:
    """
    Converts a BERT vocab.txt (one token per line, line number = id) into a JSON
    array of token strings indexed by id. Returns the number of tokens written.
    Tolerates BOM and CRLF line endings — a single trailing '\\r' on every token
    would silently make every WordPiece lookup fail and collapse output to [UNK].
    """
    with vocab_txt_path.open("r", encoding="utf-8-sig") as f:  # strips BOM if present
        tokens = [line.rstrip("\r\n") for line in f]  # tolerates LF and CRLF
    # Drop a single trailing empty token caused by a final newline, if present.
    if tokens and tokens[-1] == "":
        tokens.pop()
    _atomic_write_text(json.dumps(tokens, ensure_ascii=False), out_path)
    return len(tokens)


def bundle_assets(version: str | None, artifacts_dir: str | None, quant: str) -> None:
    """
    Copies model.onnx / vocab.json / config.json from the source artifacts into
    the app's assets/model directory under generic filenames. Prints a summary.
    """
    src = resolve_artifacts_dir(version, artifacts_dir)
    app_root = Path(__file__).resolve().parents[1] / "app" / "SymptomNerApp"
    dest = app_root / "assets" / "model"
    dest.mkdir(parents=True, exist_ok=True)

    onnx_name = "model_int8.onnx" if quant == "int8" else "model.onnx"
    onnx_src = src / onnx_name
    if not onnx_src.is_file():
        raise FileNotFoundError(f"Model file not found: {onnx_src}")

    _atomic_write_bytes(onnx_src, dest / "model.onnx")
    _atomic_write_bytes(src / "config.json", dest / "config.json")
    n_tokens = vocab_txt_to_json(src / "vocab.txt", dest / "vocab.json")

    onnx_mb = (dest / "model.onnx").stat().st_size / 1e6
    print(f"Source artifacts : {src}")
    print(f"Quantization     : {quant} ({onnx_name})")
    print(f"-> model.onnx    : {onnx_mb:.1f} MB")
    print(f"-> vocab.json    : {n_tokens} tokens")
    print(f"-> config.json   : copied")
    print(f"Destination      : {dest}")


def main() -> None:
    """Parses CLI arguments and runs the asset bundling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="v05",
        help="Artifacts version under mobile_app/artifacts/ (default: v05).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Explicit artifacts directory; overrides --version when set.",
    )
    parser.add_argument(
        "--quant",
        choices=["int8", "fp32"],
        default="int8",
        help="Which model file to bundle (default: int8).",
    )
    args = parser.parse_args()
    bundle_assets(args.version, args.artifacts_dir, args.quant)


if __name__ == "__main__":
    main()
