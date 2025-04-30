import streamlit as st
import re
import os
import gdown
import pickle

# Google Drive direct download links
model_url = 'https://drive.google.com/uc?id=1_dFRVVt6RQyNNCtcMKOiWfqIjEo1rwal'
vectorizer_url = 'https://drive.google.com/uc?id=1vvd_j6v-TNhoHhzyNYP9aY-NG_V6jdEi'

# Filenames
model_file = 'spam_model.pkl'
vectorizer_file = 'vectorizer.pkl'

# Download if not already present
if not os.path.exists(model_file):
    gdown.download(model_url, model_file, quiet=False)
if not os.path.exists(vectorizer_file):
    gdown.download(vectorizer_url, vectorizer_file, quiet=False)

# Load model and vectorizer
with open(model_file, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_file, 'rb') as f:
    vectorizer = pickle.load(f)

# Simple text preprocessing (no nltk)
def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.split()
    return ' '.join(text)

# --- Sidebar ---
st.sidebar.title("📘 About This App")
st.sidebar.markdown("""
This mini-project uses a **Machine Learning model** to detect whether a message is **Spam** or **Ham (Not Spam)**.  
**Technologies used**:  
- Streamlit  
- Scikit-learn  
- TF-IDF Vectorizer  
- Naive Bayes Classifier  

👨‍💻 Created by: *Your Name Here*
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Try Sample Messages:")
sample_msgs = {
    "Congratulations! You've won a $1000 Walmart gift card.": "Spam",
    "Your meeting is scheduled at 3 PM today.": "Ham",
    "Limited offer! Buy now and get 50% off.": "Spam"
}
for msg, label in sample_msgs.items():
    st.sidebar.write(f"**{label}:** \"{msg}\"")

# --- Main Page ---
st.markdown("<h1 style='text-align: center;'>📩 Spam Message Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste a message below to check if it's <b>Spam</b> or <b>Ham</b>.</p>", unsafe_allow_html=True)

# Input box
input_sms = st.text_area("🖊️ Enter your message here:")

# Predict button
if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        st.markdown("---")
        if prediction == "spam":
            st.error("🚫 **This message is SPAM** – Looks suspicious or promotional.")
        else:
            st.success("✅ **This message is HAM** – It seems safe and legitimate.")

        st.markdown("### 🔁 Processed Text:")
        st.code(transformed_sms)
