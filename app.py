import os
import streamlit as st
import tensorflow as tf
import gdown

# =====================
# Model Download Logic
# =====================
MODEL_PATH = "gpt2_sms_classifier.keras"
FILE_ID = "1JygwDhgiAaTMF1AZCdNpltmrL-iwF3q4"  # from your Drive link
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
# Streamlit App
# =====================
st.title("GPT-2 SMS Classifier")
st.write("Enter a message to classify it as spam or ham.")

user_input = st.text_area("Your Message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message!")
    else:
        # Make prediction
        # Assuming your model outputs probabilities for two classes [ham, spam]
        pred = model.predict([user_input])
        pred_class = "Spam" if pred[0][1] > 0.5 else "Ham"
        st.write(f"Prediction: **{pred_class}**")
