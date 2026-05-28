"""
quantize_onnx.py

Quantizes the fp32 ONNX model produced by export_to_onnx.py down to INT8.

Why quantize at all:
    - The fp32 model is ~440 MB. Apple's App Store caps cellular downloads at
      200 MB, so users on a phone network couldn't install the app. INT8 brings
      this down to ~110 MB, which fits.
    - INT8 inference is also typically faster on phone CPUs because there's
      less data to move and modern ARM chips have INT8 instructions.

What it costs:
    - A small accuracy loss. The validate_onnx.py script measures this against
      the 7-HDA evaluation set. Acceptance criterion: <= 10 errors (V05 baseline
      with fp32 PyTorch is 8 errors, so we have a 2-error budget).
"""


from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


# Input: the fp32 ONNX file produced by export_to_onnx.py
FP32_MODEL_PATH = Path("mobile_app/artifacts/v05/model.onnx")
# Output: the smaller INT8 file
INT8_MODEL_PATH = Path("mobile_app/artifacts/v05/model_int8.onnx")


def quantize_model_to_int8() -> None:
    """
    Read the fp32 ONNX model and write an INT8-quantized version next to it.

    Uses *dynamic* quantization, which:
      - quantizes weights to INT8 ahead of time (this is what shrinks the file)
      - keeps activations in fp32 and quantizes them on-the-fly at inference time
      - needs NO calibration dataset (unlike static quantization, which would
        require us to feed example inputs through the model)
    This is the standard recipe for transformer encoders.

    TODO: Make the quantization type a parameter for flexibility and experimentation purposes (LOW PRIORITY)
    """

    # Make sure the fp32 model exists. If it doesn't, the user forgot to run
    # export_to_onnx.py first.
    if not FP32_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {FP32_MODEL_PATH}. "
            f"Run `python -m mobile_app.model_prep.export_to_onnx` first."
        )

    # Run the quantization.
    #
    # model_input  : path to the fp32 .onnx file
    # model_output : path to write the INT8 .onnx file
    # weight_type  : QuantType.QInt8 means "quantize weights to signed 8-bit ints".
    #                QUInt8 (unsigned) is another option; QInt8 is the safer
    #                default for transformer weights, which are roughly
    #                symmetric around zero.
    quantize_dynamic(
        model_input=str(FP32_MODEL_PATH),
        model_output=str(INT8_MODEL_PATH),
        weight_type=QuantType.QInt8,
    )

    # Quick on-the-spot size report so you can eyeball that it worked.
    # .stat() shows file metadata so .st_size will show the size in bytes divide by 2^20 (Megabyte)
    
    fp32_mb = FP32_MODEL_PATH.stat().st_size / (2**20)
    int8_mb = INT8_MODEL_PATH.stat().st_size / (2**20)
    print(f"fp32 model: {fp32_mb:.1f} MB")
    print(f"INT8 model: {int8_mb:.1f} MB  ({fp32_mb / int8_mb:.1f}x smaller)")
    print(f"Wrote INT8 model to: {INT8_MODEL_PATH}")


if __name__ == "__main__" :
  # How to run and verify 
  # From the project root:
  # python -m mobile_app.model_prep.quantize_onnx
  quantize_model_to_int8()