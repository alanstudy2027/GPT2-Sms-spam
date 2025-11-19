import os
import streamlit as st
import tensorflow as tf
import gdown
import numpy as np
from transformers import GPT2Tokenizer

# =====================
# Model Download Logic
# =====================
MODEL_PATH = "gpt2_sms_classifier.keras"
FILE_ID = "1JygwDhgiAaTMF1AZCdNpltmrL-iwF3q4"  # Google Drive file ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# =====================
# Load Model
# =====================
model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# Load GPT-2 Tokenizer
# =====================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
MAX_LEN = 128  # should match model's input length

# =====================
# Streamlit App
# =====================
st.title("GPT-2 SMS Classifier")
st.write("Enter a message to classify it as Spam or Ham.")

user_input = st.text_area("Your Message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # -----------------------------
        # Tokenize user input
        # -----------------------------
        tokens = tokenizer.encode(user_input, add_special_tokens=True)

        # Pad or truncate to MAX_LEN
        if len(tokens) < MAX_LEN:
            tokens = tokens + [0] * (MAX_LEN - len(tokens))  # pad with 0
        else:
            tokens = tokens[:MAX_LEN]

        input_ids = np.array([tokens])  # shape = (1, MAX_LEN)

        # -----------------------------
        # Make prediction
        # -----------------------------
        pred = model.predict(input_ids)
        pred_class = "Spam" if pred[0][1] > 0.5 else "Ham"

        st.success(f"Prediction: **{pred_class}**")
