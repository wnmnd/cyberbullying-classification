import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the TFLite model
def load_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

# Initialize tokenizer
tokenizer = Tokenizer()

# Text preprocessing function
def preprocess_text(text, max_length=20):  # Changed max_length to 20 as per error message
    # Clean the text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    return padded_sequence.astype(np.float32)

# Perform prediction
def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input details for debugging
    print(f"Expected input shape: {input_details[0]['shape']}")
    print(f"Provided input shape: {input_data.shape}")
    
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    return prediction

# Mapping labels
LABELS = ['Not Cyberbullying', 'Gender', 'Religion', 'Age', 'Ethnicity']

# Streamlit App
def main():
    st.set_page_config(page_title="Cyberbullying Detection", page_icon="🔍", layout="wide")

    # Header
    st.title("🔍 Cyberbullying Detection App")
    st.markdown(
        """
        Welcome to the **Cyberbullying Detection App**! This tool uses a machine learning model to analyze 
        text and identify potential categories of cyberbullying.
        """
    )
    
    # Load model
    try:
        model_path = "random_forest_model.tflite"
        interpreter = load_model(model_path)
        
        # Get model details
        input_details = interpreter.get_input_details()
        st.debug(f"Model input details: {input_details}")
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Input Area
    st.markdown("### Enter Your Text Below")
    user_input = st.text_area("Type or paste a sentence:", placeholder="Enter text to analyze...")
    detect_button = st.button("🚨 Detect Cyberbullying")

    # Detect
    if detect_button:
        if user_input.strip():
            try:
                with st.spinner("Analyzing text..."):
                    # Preprocess the input
                    input_data = preprocess_text(user_input)
                    
                    # Predict
                    prediction = predict(interpreter, input_data)
                    category_index = np.argmax(prediction)
                    confidence = float(np.max(prediction))

                    # Display Results
                    st.success(f"🛡️ **Detected Category**: {LABELS[category_index]}")
                    
                    # Display confidence scores for all categories
                    st.markdown("### Confidence Scores:")
                    for label, score in zip(LABELS, prediction[0]):
                        score_percentage = float(score) * 100
                        st.progress(score_percentage / 100)
                        st.text(f"{label}: {score_percentage:.2f}%")

                    # Show recommendations if cyberbullying is detected
                    if category_index != 0:  # If not "Not Cyberbullying"
                        st.warning("### Recommendations:")
                        st.markdown("""
                            1. 🚫 Do not respond to the message
                            2. 📸 Save evidence (screenshots)
                            3. 🔒 Block the sender
                            4. 📢 Report to relevant authorities
                            5. 💬 Talk to a trusted person
                        """)
                        
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.error("Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        ### Disclaimer
        This tool is for educational purposes only. Please use it responsibly and seek professional advice 
        when dealing with serious cases of cyberbullying.
        """
    )

if __name__ == "__main__":
    main()
