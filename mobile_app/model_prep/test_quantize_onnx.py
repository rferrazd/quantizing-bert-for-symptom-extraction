import onnxruntime as ort
from transformers import AutoTokenizer

session = ort.InferenceSession("mobile_app/artifacts/v05/model_int8.onnx")
tokenizer = AutoTokenizer.from_pretrained("mobile_app/artifacts/v05")

inputs = tokenizer("Patient reports headache.", return_tensors="np")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
outputs = session.run(None, dict(inputs))
print("Logits shape:", outputs[0].shape)   # Expect: (1, 6, 5)