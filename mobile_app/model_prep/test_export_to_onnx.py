# Load the ONNX model and run one forward pass to confirm it works.
import onnxruntime as ort
from transformers import AutoTokenizer

# An InferenceSession is the ONNX equivalent of "loading the model into memory".
# Once created, you can call .run() on it many times.
session = ort.InferenceSession("mobile_app/artifacts/v05/model.onnx")

# Load the tokenizer that was copied next to the model.
tokenizer = AutoTokenizer.from_pretrained("mobile_app/artifacts/v05")

# Tokenize a short test sentence. return_tensors="np" gives us numpy arrays,
# which is what ONNX Runtime wants (PyTorch tensors would also work but numpy
# is the "native" choice for ONNX).
inputs = tokenizer("Patient reports headache.", return_tensors="np")

# Run the model. The first argument is "which outputs to return" — None means
# "return all outputs". The second is the input dict.
outputs = session.run(None, dict(inputs))

# outputs[0] is the logits tensor. Shape should be (1, sequence_length, 5).
# 1 = batch size, sequence_length = number of tokens, 5 = number of labels
# (O, B-SYMPTOM_POS, I-SYMPTOM_POS, B-SYMPTOM_NEG, I-SYMPTOM_NEG).
print("Logits shape:", outputs[0].shape)
# Expect something like: Logits shape: (1, 7, 5) -- test passed