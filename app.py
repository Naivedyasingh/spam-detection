import streamlit as st
import joblib
import re
import string
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing Utilities
stemmer = PorterStemmer()
HTMLTAGS = re.compile('<.*?>')
PUNCT_TABLE = str.maketrans('', '', string.punctuation)
DIGIT_TABLE = str.maketrans('', '', string.digits)
MULTIPLE_WHITESPACE = re.compile(r"\s+")

def preprocess_text(text):
    text = HTMLTAGS.sub('', text)  # Remove HTML tags
    text = text.translate(PUNCT_TABLE)  # Remove punctuation
    text = text.translate(DIGIT_TABLE)  # Remove digits
    text = text.lower()
    text = MULTIPLE_WHITESPACE.sub(" ", text).strip()
    words = text.split()
    stemmed = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed)

# --- Sidebar ---
st.sidebar.title("📘 About This App")
st.sidebar.markdown("""
This mini project uses a **Machine Learning model** to predict the sentiment of Amazon product reviews.  
**Technologies used**:  
- Streamlit  
- NLTK  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression / Naive Bayes (as used in your model)

👨‍💻 Created by: *Your Name Here*
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Try Sample Reviews:")
sample_reviews = {
    "Great product! Works perfectly.": "Positive",
    "Terrible experience. Do not buy.": "Negative",
    "It’s okay, nothing special.": "Neutral"
}
for example, label in sample_reviews.items():
    st.sidebar.write(f"**{label}:** \"{example}\"")

# --- Main Page ---
st.markdown("<h1 style='text-align: center;'>📝 Amazon Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste your review below to find out if it's <b>Positive</b>, <b>Negative</b>, or <b>Neutral</b>!</p>", unsafe_allow_html=True)

# Input box
user_input = st.text_area("🖊️ Enter your Amazon review here:")

# Predict button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        clean_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]

        st.markdown("---")
        if prediction == 'Positive':
            st.success("✅ **Sentiment: Positive** – This review seems satisfied and appreciative!")
        elif prediction == 'Negative':
            st.error("❌ **Sentiment: Negative** – This review indicates dissatisfaction.")
        else:
            st.info("➖ **Sentiment: Neutral** – The review is neither strongly positive nor negative.")

        st.markdown("### 🔁 Processed Text:")
        st.code(clean_text)
