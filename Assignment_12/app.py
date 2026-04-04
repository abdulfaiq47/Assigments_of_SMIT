import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

# Set Page Config
st.set_page_config(page_title="Next Word Predictor", layout="centered")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
model_path = os.path.join(BASE_DIR, "next_word_model.h5")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Load model and tokenizer (Cached to prevent reloading on every click)
@st.cache_resource
def load_assets():
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    max_sequence_len = 50

    # Streamlit UI
    st.title("🔮 Next Word Predictor")
    st.write("Enter a phrase below to predict the next word.")

    input_text = st.text_input("Enter your text:", placeholder="Type something...")
    top_k = st.slider("Number of predictions", 1, 5, 1)

    if st.button("Predict"):
        if input_text:
            sequence = tokenizer.texts_to_sequences([input_text])[0]
            if len(sequence) > 0:
                token_list = pad_sequences([sequence], maxlen=max_sequence_len - 1, padding="pre")
                predicted_probs = model.predict(token_list, verbose=0)[0]

                top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
                next_words = [index_word.get(i) for i in top_indices if index_word.get(i)]

                st.success(f"Suggestions: **{', '.join(next_words)}**")
            else:
                st.warning("Please enter a valid word recognized by the tokenizer.")
        else:
            st.error("Please enter some text first!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Check if next_word_model.h5 and tokenizer.pkl are in the Assignment_12 folder.")
