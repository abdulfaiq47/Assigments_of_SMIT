from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
model_path = os.path.join(BASE_DIR, "next_word_model.h5")  # use your .h5 file
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Load model
model = load_model(model_path)

# Load tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Reverse mapping
index_word = {v: k for k, v in tokenizer.word_index.items()}

# Maximum sequence length
max_sequence_len = 50  # must match what you used during training

@app.route("/api/test")
def test():
    return jsonify({"message": "API working fine!!!"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    top_k = int(data.get("top_k", 1))

    sequence = tokenizer.texts_to_sequences([text])[0]
    if len(sequence) == 0:
        return jsonify({"next_words": []})

    token_list = pad_sequences([sequence], maxlen=max_sequence_len - 1, padding="pre")
    predicted_probs = model.predict(token_list, verbose=0)[0]

    top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    next_words = [index_word.get(i) for i in top_indices if index_word.get(i)]

    return jsonify({"next_words": next_words})

if __name__ == "__main__":
    # Streamlit cloud / Heroku friendly
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
