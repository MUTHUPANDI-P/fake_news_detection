import streamlit as st
from transformers import pipeline
import torch
import gdown
import os

# --- Download model from Google Drive ---
# Replace with your Google Drive FILE_ID
FILE_ID = "1dn8EeVaDUVYtqspyHrIT24f5Vo0pFsWH"  # Example: '1AbCdEfGhIjKlMnOpQrStUvWxYz'

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

# Create folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download only if model not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# --- Load Hugging Face pipeline ---
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, device=device)

# --- Streamlit UI ---
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

st.markdown("<h1 style='text-align: center;'>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Enter a news headline or short article to check if it's Real or Fake.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input text
input_text = st.text_area("📝 Enter news headline or article snippet:", height=150)

# Predict button
if st.button("🔍 Predict"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = pipe(input_text)[0]
            label = result['label']
            confidence = result['score']

            # Display result
            label_display = "✅ Real News" if label == "LABEL_1" else "❌ Fake News"
            label_color = "green" if label == "LABEL_1" else "red"

            st.markdown(f"<h3 style='color: {label_color}; text-align: center;'>{label_display}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}</b></p>", unsafe_allow_html=True)

            st.markdown("---")
            with st.expander("📊 Model Info"):
                st.write("• Model: Fine-tuned DistilBERT")
                st.write(f"• Confidence: {confidence:.4f}")
                st.write("• LABEL_1 = Real News")
                st.write("• LABEL_0 = Fake News")
