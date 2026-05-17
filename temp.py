from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# Download and convert
model = ORTModelForFeatureExtraction.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Save locally
model.save_pretrained("./model/all-MiniLM-L6-v2-onnx")
tokenizer.save_pretrained("./model/all-MiniLM-L6-v2-onnx")