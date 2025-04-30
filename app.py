import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open('model2.pkl', 'rb'))  # Must be pre-trained
vectorizer = pickle.load(open('vectorizer2.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>📩 Email/SMS Spam Classifier</h1>
    <p style='text-align: center; font-size: 18px;'>Instantly detect spam messages using a trained ML model.</p>
    <hr style="border:1px solid #f0f0f0;">
""", unsafe_allow_html=True)

# Input area
message = st.text_area("💬 Enter your message below:", height=150)

# Predict button
if st.button('🚀 Predict'):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocess and predict
        transformed_message = transform_text(message)
        vector_input = vectorizer.transform([transformed_message])
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1 or result == 'spam':
            st.error("🚫 This message is **SPAM**!")
        else:
            st.success("✅ This message is **HAM** (not spam).")

        st.balloons()

# Footer
st.markdown("""
    <hr style="border:1px solid #f0f0f0;">
    <p style='text-align: center; font-size: 14px;'>Built with ❤️ using Streamlit & Scikit-learn</p>
""", unsafe_allow_html=True)
