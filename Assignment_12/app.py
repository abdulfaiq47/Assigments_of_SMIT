# app.py
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "next_word_model_fixed.keras")  # your saved model
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")      # your tokenizer
MAX_SEQUENCE_LEN = 50  # same as in training

# --- LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- PREDICTION FUNCTION ---
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return None
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    prediction = model.predict(token_list, verbose=0)
    prediction_word_index = np.argmax(prediction)
    for word, index in tokenizer.word_index.items():
        if index == prediction_word_index:
            return word
    return None

# --- STREAMLIT UI ---
st.title("Next Word Predictor")
st.write("Type some text and I will predict the next word for you!")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        next_word = predict_next_word(model, tokenizer, user_input, MAX_SEQUENCE_LEN)
        if next_word:
            st.success(f"Next word prediction: **{next_word}**")
        else:
            st.error("Could not predict the next word. Try adding more context.")