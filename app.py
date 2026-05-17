from flask import Flask, request, render_template
import numpy as np
import pickle
from numpy.linalg import norm
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

app = Flask(__name__)

# Load data
with open("model/embeddings.pkl", "rb") as f:
    model_data = pickle.load(f)

embeddings = model_data['embeddings']
disease_names = model_data['disease_names']
source_urls = model_data['source_urls']
sentences = model_data['sentences']

# Load ONNX model and tokenizer
session = InferenceSession("model/all-MiniLM-L6-v2-onnx/model.onnx")
tokenizer = AutoTokenizer.from_pretrained("model/all-MiniLM-L6-v2-onnx")

def get_embedding(text):
    """Get embedding using ONNX model"""
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"]  # Add this line
    })
    
    embedding = outputs[0].mean(axis=1)
    return embedding
    

def predict_disease(user_input):
    user_embedding = get_embedding(user_input)
    
    # Cosine similarity using numpy
    similarities = np.dot(embeddings, user_embedding.T) / (norm(embeddings, axis=1, keepdims=True) * norm(user_embedding))
    similarities = similarities.flatten()
    
    top_idx = np.argmax(similarities)
    top_score = similarities[top_idx]

    return disease_names[top_idx], source_urls[top_idx], sentences[top_idx], top_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    weight = float(request.form['weight'])
    height_cm = float(request.form['height'])
    
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)
    
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obesity"

    disease, url, sentence, score = predict_disease(symptoms)

    prediction_text = f"<strong>Disease:</strong> {disease}<br>" \
                      f"<strong>Matched Sentence:</strong> {sentence}<br>" \
                      f"<strong>Confidence Score:</strong> {score:.2f}<br>" \
                      f"<strong>Source:</strong> <a href='{url}' target='_blank'>{url}</a>"
    bmi_text = f"<strong>Your BMI:</strong> {bmi:.1f} ({bmi_category})"
    
    return render_template('index.html', prediction_text=prediction_text, bmi_text=bmi_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)