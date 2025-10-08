import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# ------------------------------
# 🎨 PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="MindScope - Mental Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# 🧠 HEADER
# ------------------------------
st.title("🧠 MindScope: AI Mental Health Sentiment Analyzer")
st.markdown("""
Gain insights into emotional and mental health indicators using deep learning.
Analyze journal entries, chats, or text inputs to detect emotional patterns.
""")

# ------------------------------
# 🧩 LOAD MODEL AND TOKENIZER
# ------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
max_len = 100  # must match training

# Define your custom categories
labels = [
    "Normal 🙂",
    "Depression 😞",
    "Suicidal ⚠️",
    "Anxiety 😰",
    "Stress 😫",
    "Bi-Polar 😶‍🌫️",
    "Personality Disorder 🧩"
]

# ------------------------------
# 🎛️ SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.header("📊 Navigation")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["Single Text Analysis", "Batch CSV Upload", "About App"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed BY ARAVIND REDDY G**")

# ------------------------------
# ✳️ SINGLE TEXT MODE
# ------------------------------
if mode == "Single Text Analysis":
    st.subheader("💬 Enter Text for Analysis")
    user_input = st.text_area("Type your thoughts, diary entry, or sentence here:")

    if st.button("🔍 Analyze"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text before analyzing.")
        else:
            with st.spinner("Analyzing... please wait ⏳"):
                seq = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=max_len)
                preds = model.predict(padded)
                label_index = np.argmax(preds)
                confidence = np.max(preds) * 100
                predicted_label = labels[label_index]

            # Display Results
            st.success(f"**Predicted Category:** {predicted_label}")
            st.caption(f"Confidence: {confidence:.2f}%")

            # Bar Chart of All Probabilities
            st.subheader("Category Confidence Distribution:")
            probs = {labels[i]: float(preds[0][i]) for i in range(len(labels))}
            st.bar_chart(probs)

            # Mood Indicator
            if "Normal" in predicted_label:
                st.info("✅ Emotional state appears balanced and stable.")
            elif "Depression" in predicted_label:
                st.warning("💡 Signs of sadness or low mood detected.")
            elif "Suicidal" in predicted_label:
                st.error("🚨 Critical indicators detected! Seek professional help immediately.")
            elif "Anxiety" in predicted_label:
                st.warning("⚠️ Elevated stress or anxiety indicators detected.")
            elif "Stress" in predicted_label:
                st.warning("😓 Signs of psychological stress detected.")
            elif "Bi-Polar" in predicted_label:
                st.info("🔄 Possible mood fluctuation patterns found.")
            elif "Personality" in predicted_label:
                st.info("🧩 Indicators of distinct personality traits observed.")

# ------------------------------
# 📁 BATCH CSV MODE
# ------------------------------
elif mode == "Batch CSV Upload":
    st.subheader("📄 Upload a CSV file for batch sentiment analysis")
    st.markdown("Your file should have a column named **'text'** containing the entries to analyze.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ CSV must contain a column named 'text'.")
        else:
            with st.spinner("Processing all entries... ⏳"):
                sequences = tokenizer.texts_to_sequences(df["text"].astype(str))
                padded = pad_sequences(sequences, maxlen=max_len)
                preds = model.predict(padded)

                df["Predicted_Label"] = [labels[np.argmax(p)] for p in preds]
                df["Confidence (%)"] = [round(np.max(p) * 100, 2) for p in preds]
                df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.success("✅ Analysis completed successfully!")
            st.dataframe(df)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="mental_health_analysis_results.csv",
                mime="text/csv",
            )

# ------------------------------
# ℹ️ ABOUT SECTION
# ------------------------------
elif mode == "About App":
    st.header("ℹ️ About MindScope")
    st.markdown("""
**MindScope** is an AI-powered application that analyzes textual content for
indicators of mental health conditions such as depression, anxiety, or stress.  
It uses a deep learning **LSTM** model trained on mental health–related data.

""")

