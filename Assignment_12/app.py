import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

st.set_page_config(page_title="Next Word Predictor")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "next_word_model.h5")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

@st.cache_resource
def load_my_assets():
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

st.title("🔮 Next Word Predictor")

try:
    model, tokenizer = load_my_assets()
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    
    input_text = st.text_input("Enter your phrase:")
    if st.button("Predict") and input_text:
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([sequence], maxlen=49, padding="pre")
        predicted = model.predict(token_list, verbose=0)[0]
        next_word = index_word.get(np.argmax(predicted))
        st.success(f"Predicted next word: **{next_word}**")

except Exception as e:
    st.error(f"File Error: {e}")
    st.info("If you see 'file signature not found', please do Step 2 below.")
