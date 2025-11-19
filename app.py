import os
import streamlit as st
import tensorflow as tf
import gdown
from transformers import GPT2TokenizerFast
import numpy as np

# =====================
# Model Download Logic
# =====================
MODEL_PATH = "gpt2_sms_classifier.keras"
FILE_ID = "1JygwDhgiAaTMF1AZCdNpltmrL-iwF3q4"  # Google Drive file ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# =====================
# Load Model & Tokenizer
# =====================
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
MAX_LEN = 128  # Same as model training

# =====================
# Streamlit UI
# =====================
st.title("GPT-2 SMS Classifier")
st.write("Enter a message to classify it as spam or ham.")

user_input = st.text_area("Your Message:")

def preprocess(text):
    tokens = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="tf"
    )
    return tokens["input_ids"]

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        input_ids = preprocess(user_input)
        
        # Make prediction safely for Streamlit/M1
        pred = model(input_ids, training=False).numpy()
        pred_class = "Spam" if pred[0][1] > 0.5 else "Ham"
        
        st.success(f"Prediction: **{pred_class}**")
