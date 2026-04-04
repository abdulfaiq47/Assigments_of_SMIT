input_text = st.text_input("Enter your phrase:")
    
    # Add a slider so you can choose how many words to see
    top_k = st.slider("Number of predictions", 1, 5, 3)

    if st.button("Predict") and input_text:
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        
        if len(sequence) > 0:
            token_list = pad_sequences([sequence], maxlen=49, padding="pre")
            
            # Get probabilities for all words in the vocabulary
            predicted_probs = model.predict(token_list, verbose=0)[0]
            
            # Get the indices of the top_k highest probabilities
            # argsort sorts ascending, so we take the last k elements and reverse them
            top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
            
            # Map indices back to words
            next_words = [index_word.get(idx) for idx in top_indices if index_word.get(idx)]
            
            st.write("### Suggestions:")
            for i, word in enumerate(next_words, 1):
                st.success(f"{i}. **{word}**")
        else:
            st.warning("The words entered are not in the model's vocabulary.")
