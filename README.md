# 🧬 Disease Predictor — NLP

> Describe your symptoms in plain English. Get an instant disease prediction powered by semantic search and sentence embeddings.

**[🚀 Live Demo](https://disease-predictor-pbrr.onrender.com/)**

---

## What It Does

Disease Predictor takes a free-text description of your symptoms, converts it into a semantic vector using a lightweight NLP model, and finds the closest matching disease from a pre-built knowledge base — all without any keyword rules or hard-coded logic.

It also calculates your **BMI** on the fly using your weight and height, giving you additional health context alongside the prediction.

---

## How It Works

```
User types symptoms
       ↓
Tokenized by all-MiniLM-L6-v2 tokenizer
       ↓
Passed through ONNX inference session
       ↓
Mean-pooled to a fixed-size embedding vector
       ↓
Cosine similarity computed against stored disease embeddings
       ↓
Top-matching disease returned with source URL + confidence score
```

### Key Components

| Component | Role |
|---|---|
| `all-MiniLM-L6-v2` (ONNX) | Converts symptom text into a 384-dim embedding |
| `embeddings.pkl` | Pre-computed disease embeddings + metadata |
| Cosine Similarity (NumPy) | Finds the closest disease match |
| Flask | Serves the web interface |
| Jinja2 Templates | Renders prediction results in HTML |

The model runs entirely via **ONNX Runtime** — no PyTorch or GPU required. Inference is fast and CPU-friendly.

---

## Project Structure

```
Disease-predictor-NLP/
├── app.py                         # Flask app — routes, prediction logic, BMI calc
├── requirements.txt               # Python dependencies
├── model/
│   ├── embeddings.pkl             # Disease embeddings, names, sentences, source URLs
│   └── all-MiniLM-L6-v2-onnx/
│       ├── model.onnx             # Quantized sentence transformer model
│       └── tokenizer files        # HuggingFace tokenizer config
└── templates/
    └── index.html                 # Frontend — symptom form + results display
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/RohitThakur2006/Disease-predictor-NLP.git
cd Disease-predictor-NLP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open your browser at **`http://localhost:5000`**

---

## Usage

1. Open the app in your browser
2. Enter your symptoms in plain English — e.g. *"fever, cough, and difficulty breathing"*
3. Enter your weight (kg) and height (cm)
4. Hit **Predict**

The app will return:
- **Predicted Disease** — the closest semantic match
- **Matched Sentence** — the specific description that matched your input
- **Confidence Score** — cosine similarity score (0–1)
- **Source URL** — the reference where the disease info was sourced from
- **BMI** — your body mass index with weight category

---

## Dependencies

```
Flask==3.0.2
numpy==1.24.3
onnxruntime==1.16.0
transformers==4.30.0
```

---

## Disclaimer

> This tool is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.

---

## Author

**Rohit Thakur** — [GitHub](https://github.com/RohitThakur2006)
