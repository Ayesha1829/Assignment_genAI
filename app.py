import streamlit as st
import pandas as pd
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ----------------------------
# Load Pre-trained Model
# ----------------------------
try:
    model = load_model('lstm_poetry_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

# ----------------------------
# Load Dataset and Prepare Tokenizer
# ----------------------------
try:
    df = pd.read_csv("Roman-Urdu-Poetry.csv")
    df.columns = df.columns.str.strip()  # Remove any hidden spaces in column names
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'Roman-Urdu-Poetry.csv' is in the same directory.")

# ----------------------------
# Text Cleaning Function
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Zāčēğīñōūṣṭẓ ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# Prepare Tokenizer
# ----------------------------
if "Poetry" in df.columns:
    poetry_texts = df["Poetry"].apply(clean_text).tolist()
    tokenizer = Tokenizer(num_words=7000, oov_token="<OOV>")
    tokenizer.fit_on_texts(poetry_texts)
else:
    st.error("The dataset does not contain a 'Poetry' column.")
    tokenizer = None

# Set the sequence length (should match what was used during training)
max_seq_length = 10

# ----------------------------
# Utility Functions for Generation
# ----------------------------
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature  # Apply temperature scaling
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)  # Normalize to get probabilities
    return np.random.choice(len(preds), p=preds)  # Randomly sample an index

def generate_poetry(seed_text, next_words=30, temperature=0.8):
    global model  # Ensure we reference the global model variable
    if not seed_text or len(seed_text.strip()) == 0:
        return "Please provide a valid seed text."
    
    seed_text = clean_text(seed_text)
    
    # (Re)fit tokenizer on the dataset if needed.
    if "Poetry" in df.columns:
        tokenizer.fit_on_texts(df['Poetry'].astype(str).tolist())
    else:
        return "❌ Column 'Poetry' not found in dataset."
    
    # Generate words one at a time.
    for _ in range(next_words):
        tokenized_input = tokenizer.texts_to_sequences([seed_text])
        tokenized_input = pad_sequences(tokenized_input, maxlen=max_seq_length, padding='pre')
        
        if len(tokenized_input) == 0 or len(tokenized_input[0]) == 0:
            return "Error: Input text is not in vocabulary."
        
        predicted_probs = model.predict(tokenized_input)[0]
        predicted_index = sample_with_temperature(predicted_probs, temperature)
        predicted_word = tokenizer.index_word.get(predicted_index, "")
        if not predicted_word:
            break
        seed_text += " " + predicted_word
        
    return seed_text

# ----------------------------
# Streamlit User Interface
# ----------------------------
st.title("Roman Urdu Poetry Generator")
st.subheader("Generate poetic lines in Roman Urdu with AI")

seed_text_input = st.text_input("Enter a seed phrase or line for poetry:", "")
temperature_slider = st.slider("Select temperature (0.0 - 1.5)", 0.0, 1.5, 0.8, 0.1)
next_words_slider = st.slider("Number of words to generate", 5, 50, 30)

if st.button("Generate Poetry"):
    if seed_text_input:
        poetry_output = generate_poetry(seed_text_input, next_words_slider, temperature_slider)
        st.subheader("Generated Poetry:")
        st.write(poetry_output)
    else:
        st.error("Please enter a seed phrase to generate poetry.")
