import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

# Set Page Config
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

# Base directory for pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "next_word_model.h5")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Cache model loading for speed
@st.cache_resource
def load_assets():
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# UI Header
st.title("🔮 Next Word Predictor")
st.markdown("---")

try:
    model, tokenizer = load_assets()
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    max_sequence_len = 49 # Adjusted for padding logic

    # User Input
    input_text = st.text_input("Enter your phrase:", placeholder="e.g. How are you")
    top_k = st.slider("Number of suggestions", 1, 5, 3)

    if st.button("Predict"):
        if input_text.strip():
            # Convert text to sequence
            sequence = tokenizer.texts_to_sequences([input_text])[0]
            
            if len(sequence) > 0:
                # Pad sequence
                token_list = pad_sequences([sequence], maxlen=max_sequence_len, padding="pre")
                
                # Get prediction probabilities
                predicted_probs = model.predict(token_list, verbose=0)[0]
                
                # Get Top K indices
                top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
                
                # Map to words
                next_words = [index_word.get(idx) for idx in top_indices if index_word.get(idx)]
                
                # Display results
                st.subheader("Top Suggestions:")
                for i, word in enumerate(next_words, 1):
                    st.success(f"{i}. **{word}**")
            else:
                st.warning("Words not found in vocabulary. Try different words.")
        else:
            st.error("Please enter some text first!")

except Exception as e:
    st.error(f"Execution Error: {e}")
    st.info("Ensure 'next_word_model.h5' and 'tokenizer.pkl' are in the same folder as this script.")

st.markdown("---")
st.caption("Developed for Assignment 12")
