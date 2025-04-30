import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('model2.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer2.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            y.append(ps.stem(word))

    return " ".join(y)

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.markdown("Enter a message below to check if it's **Spam** or **Not Spam**.")

# Input from user
input_sms = st.text_area("Enter the message here")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = vectorizer.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display result
    if result == 1:
        st.error("🚫 This message is **Spam**.")
    else:
        st.success("✅ This message is **Not Spam**.")
