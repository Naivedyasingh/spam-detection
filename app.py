import streamlit as st
import nltk
import gdown
import pickle
import os

# Download NLTK data
nltk.download('punkt', quiet=True)

# Google Drive direct download links
model_url = 'https://drive.google.com/uc?id=1_dFRVVt6RQyNNCtcMKOiWfqIjEo1rwal'
vectorizer_url = 'https://drive.google.com/uc?id=1vvd_j6v-TNhoHhzyNYP9aY-NG_V6jdEi'

# Filenames to save downloaded files
model_file = 'spam_model.pkl'
vectorizer_file = 'vectorizer.pkl'

# Download model and vectorizer if not already present
if not os.path.exists(model_file):
    gdown.download(model_url, model_file, quiet=False)

if not os.path.exists(vectorizer_file):
    gdown.download(vectorizer_url, vectorizer_file, quiet=False)

# Load the model and vectorizer
with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_file, 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    return ' '.join(text)

# Streamlit UI
st.title("📩 Spam Detection App")
input_sms = st.text_area("Enter message to classify")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]

        if prediction == "spam":
            st.error("🚫 This message is SPAM")
        else:
            st.success("✅ This message is HAM (Not Spam)")
