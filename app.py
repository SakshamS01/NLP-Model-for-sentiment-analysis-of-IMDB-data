
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (only if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4', quiet=True)

# Load the saved components
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')
best_sentiment_model = joblib.load('best_sentiment_model.joblib')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text Cleaning Function (as defined in the notebook)
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization and Stopword Removal Function (as defined in the notebook)
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Lemmatization Function (as defined in the notebook)
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

# --- Streamlit App --- 
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below to get its sentiment (positive/negative).")

user_input = st.text_area("Enter Review Here:", "This movie was absolutely fantastic! I loved every moment of it.")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_text = clean_text(user_input)
        processed_text = tokenize_and_remove_stopwords(cleaned_text)
        lemmatized_text = lemmatize_text(processed_text)

        # Transform text using the loaded TF-IDF vectorizer
        input_tfidf = tfidf_vectorizer.transform([lemmatized_text])

        # Make prediction
        prediction = best_sentiment_model.predict(input_tfidf)

        # Decode prediction
        sentiment_label = label_encoder.inverse_transform(prediction)[0]

        st.subheader("Prediction:")
        if sentiment_label == 'positive':
            st.success(f"The sentiment of the review is: **{sentiment_label.upper()}** 😊")
        else:
            st.error(f"The sentiment of the review is: **{sentiment_label.upper()}** 😠")
    else:
        st.warning("Please enter a review to analyze.")
