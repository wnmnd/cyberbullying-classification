import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Set page config
st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .category-label {
        font-weight: bold;
        color: #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare the model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="random_forest_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Text preprocessing function
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Initialize the tokenizer
tokenizer = Tokenizer()

# Categories
categories = ['Not Cyberbullying', 'Gender', 'Religion', 'Age', 'Ethnicity']

def main():
    # Header
    st.title("🛡️ Cyberbullying Detection System")
    st.markdown("""
        <p style='font-size: 1.2rem; color: #666;'>
        Detect different types of cyberbullying in text messages and social media posts.
        </p>
    """, unsafe_allow_html=True)
    
    # Text input
    user_input = st.text_area(
        "Enter the text to analyze:",
        height=150,
        placeholder="Type or paste the text here..."
    )
    
    # Detect button
    if st.button("Detect", key="detect_button"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Load the model
                interpreter = load_model()
                
                # Get input and output tensors
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Preprocess the input text
                processed_text = preprocess_text(user_input)
                
                # Tokenize and pad the text (adjust max_length according to your model)
                sequence = tokenizer.texts_to_sequences([processed_text])
                padded_sequence = pad_sequences(sequence, maxlen=100)
                
                # Set the input tensor
                interpreter.set_tensor(input_details[0]['index'], padded_sequence)
                
                # Run inference
                interpreter.invoke()
                
                # Get the output tensor
                predictions = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = np.argmax(predictions[0])
                
                # Display results
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(f"### Detected Category: <span class='category-label'>{categories[predicted_class]}</span>", unsafe_allow_html=True)
                
                # Display confidence scores
                st.markdown("### Confidence Scores:")
                for category, confidence in zip(categories, predictions[0]):
                    confidence_percentage = confidence * 100
                    st.progress(confidence_percentage / 100)
                    st.text(f"{category}: {confidence_percentage:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add recommendations based on the detection
                if predicted_class != 0:
                    st.markdown("""
                        ### Recommendations:
                        1. 🚫 Do not engage with harmful content
                        2. 📸 Take screenshots as evidence
                        3. 🔒 Block the sender if possible
                        4. 📢 Report to relevant authorities
                        5. 💬 Seek support from trusted individuals
                    """)

if __name__ == "__main__":
    main()
