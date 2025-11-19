import streamlit as st
import tensorflow as tf
import numpy as np

# ------------------------------
# Load model with caching
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("gpt2_sms_classifier.keras")
    return model

model = load_model()

# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess_text(text):
    """
    Preprocess input text for GPT-2 SMS classifier.
    You may need to modify this based on how your model was trained.
    """
    # Example: simple lowercasing
    return text.lower()

# ------------------------------
# Prediction function
# ------------------------------
def predict_sms(text):
    processed_text = preprocess_text(text)
    
    # Model expects tokenized input; update if you used a tokenizer
    # Here assuming a simple example of text as input
    input_array = np.array([processed_text])
    
    # Predict
    prediction = model.predict(input_array)
    
    # Assuming binary classification: 0 = ham, 1 = spam
    label = "Spam" if np.argmax(prediction, axis=1)[0] == 1 else "Ham"
    confidence = np.max(prediction)
    return label, confidence

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="GPT-2 SMS Classifier", layout="centered")
st.title("📩 GPT-2 SMS Classifier")
st.write("Classify SMS messages as Spam or Ham!")

# Input
sms_input = st.text_area("Enter SMS message:")

# Predict button
if st.button("Predict"):
    if sms_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        label, confidence = predict_sms(sms_input)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2f}")
