import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download punkt tokenizer data (this will be done only once)
nltk.download('punkt', quiet=True)

# Pre-trained model and vectorizer (for demonstration, you can replace with your actual model)
vectorizer = TfidfVectorizer()
model = MultinomialNB()

# Example data (replace with your actual training data and model)
X_train = ["Free cash prize!", "Win a lottery", "Important update about your account", "Meeting tomorrow at 10 AM"]
y_train = ["spam", "spam", "ham", "ham"]

# Fit the vectorizer and model (normally, this should be done once, and the model should be saved)
X_train_vec = vectorizer.fit_transform(X_train)
model.fit(X_train_vec, y_train)

# Function to preprocess and tokenize text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # Tokenize text using NLTK
    return ' '.join(text)

# Streamlit UI
st.title("Spam Detection App")
input_sms = st.text_area("Enter message to classify")

if st.button('Predict'):
    # Preprocess the input
    transformed_sms = transform_text(input_sms)

    # Vectorize the input
    vector_input = vectorizer.transform([transformed_sms])

    # Predict using the trained model
    prediction = model.predict(vector_input)

    if prediction == "spam":
        st.error("This message is SPAM")
    else:
        st.success("This message is HAM (Not Spam)")

