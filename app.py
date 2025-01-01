import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import emoji

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing functions
def strip_emoji(text):
    return re.sub(emoji.get_emoji_regexp(), r"", text)

def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    text = ' '.join(word for word in text.split() if len(word) < 14)
    return text

def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2

def filter_chars(text):
    sent = []
    for word in text.split(' '):
        if ('$' in word) or ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)

def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

def preprocess_text(text):
    text = strip_emoji(text)
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    return text

# Load and preprocess training data
@st.cache_data
def load_training_data():
    df = pd.read_csv('cyberbullying_tweets.csv')
    df = df[df['cyberbullying_type'] != 'other_cyberbullying']
    df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})
    return df

# Initialize TF-IDF vectorizer
@st.cache_resource
def get_vectorizer(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(texts)
    return vectorizer

def main():
    st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            width: 100%;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Cyberbullying Detection System")
    
    # Load and preprocess data
    df = load_training_data()
    
    # Preprocess all texts
    texts_cleaned = [preprocess_text(text) for text in df['text']]
    vectorizer = get_vectorizer(texts_cleaned)
    
    # Train-test split and model training
    X = vectorizer.transform(texts_cleaned)
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sidebar
    st.sidebar.title("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Choose Classification Model",
        ["Random Forest", "Support Vector Machine", "Naive Bayes"]
    )

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text for Analysis")
        user_input = st.text_area(
            "Type or paste the text to analyze:",
            height=150,
            placeholder="Enter the text you want to analyze..."
        )
        
        if st.button("Analyze Text"):
            if user_input.strip():
                # Preprocess input text
                processed_input = preprocess_text(user_input)
                vectorized_input = vectorizer.transform([processed_input])
                
                # Make prediction
                if model_type == "Random Forest":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_type == "Support Vector Machine":
                    from sklearn.svm import LinearSVC
                    model = LinearSVC(random_state=42)
                else:
                    from sklearn.naive_bayes import MultinomialNB
                    model = MultinomialNB()
                
                # Train model and predict
                with st.spinner("Training model and analyzing text..."):
                    model.fit(X_train, y_train)
                    prediction = model.predict(vectorized_input)
                    
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(vectorized_input)
                    else:
                        # For models without predict_proba (like LinearSVC)
                        probabilities = None
                
                # Display results
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown(f"### Detected Category: **{prediction[0]}**")
                
                if probabilities is not None:
                    st.subheader("Confidence Scores:")
                    for category, prob in zip(model.classes_, probabilities[0]):
                        st.progress(prob)
                        st.text(f"{category}: {prob*100:.2f}%")
                
                # Show recommendations if cyberbullying is detected
                if prediction[0] != "not_cyberbullying":
                    st.warning("### Recommendations:")
                    st.markdown("""
                        1. 🚫 Do not engage with the content
                        2. 📸 Document the incident
                        3. 🔒 Block the sender
                        4. 📢 Report to relevant authorities
                        5. 💬 Seek support if needed
                    """)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Please enter some text to analyze.")
    
    with col2:
        st.subheader("About the Model")
        st.write(f"Currently using: **{model_type}**")
        st.write("This model analyzes text to detect different types of cyberbullying:")
        st.markdown("""
        - Gender-based
        - Religious
        - Age-based
        - Ethnic/Racial
        - Not Cyberbullying
        """)

    # Show model performance metrics if requested
    if st.sidebar.checkbox("Show Model Performance Metrics"):
        st.subheader("Model Performance Metrics")
        with st.spinner("Calculating metrics..."):
            y_pred = model.predict(X_test)
            
            # Classification Report
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.write("Classification Report:")
            st.dataframe(df_report)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
