import os
import streamlit as st
import tensorflow as tf
import gdown
from transformers import GPT2Tokenizer

# =====================
# Model Download Logic
# =====================
MODEL_PATH = "gpt2_sms_classifier.keras"
FILE_ID = "1JygwDhgiAaTMF1AZCdNpltmrL-iwF3q4"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# =====================
# Load Model + Tokenizer
# =====================
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# =====================
# Streamlit App
# =====================
st.title("GPT-2 SMS Classifier")
st.write("Enter a message to classify it as spam or ham.")

user_input = st.text_area("Your Message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # -----------------------------
        # Tokenize input with GPT2 tokenizer
        # -----------------------------
        input_ids = tokenizer.encode(user_input, return_tensors="tf", truncation=True, max_length=128)

        # -----------------------------
        # Make prediction
        # -----------------------------
        pred = model(input_ids)
        pred_class = "Spam" if pred[0][1] > 0.5 else "Ham"

        st.success(f"Prediction: **{pred_class}**")
