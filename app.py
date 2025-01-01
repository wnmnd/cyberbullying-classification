import streamlit as st
import numpy as np
import joblib

# Load the joblib model
def load_model(joblib_path):
    try:
        model = joblib.load(joblib_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess text (You may need to adapt this based on your model's input requirements)
def preprocess_text(text, vectorizer):
    try:
        transformed_text = vectorizer.transform([text])
        return transformed_text
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return None

# Mapping labels
LABELS = ['Religion', 'Age', 'Gender', 'Ethnicity', 'Not Bullying']

# Streamlit App
def main():
    st.set_page_config(page_title="Cyberbullying Detection", page_icon="🔍", layout="wide")

    # Header
    st.title("🔍 Cyberbullying Detection App")
    st.markdown(
        """
        Welcome to the **Cyberbullying Detection App**! This tool uses a machine learning model to analyze 
        text and identify potential categories of cyberbullying. It is designed to assist in understanding 
        harmful online behavior and promote a safer digital environment.
        """
    )
    st.markdown("---")

    # Description Section
    st.markdown(
        """
        ### How It Works
        1. Enter any text in the input box below. 
        2. Click on the **Detect Cyberbullying** button. 
        3. The app will classify the text into one of the following categories:
           - **Religion-based Cyberbullying**
           - **Age-based Cyberbullying**
           - **Gender-based Cyberbullying**
           - **Ethnicity-based Cyberbullying**
           - **Not Bullying**

        ### Why This Matters
        Identifying cyberbullying is crucial in combating harmful online interactions. 
        By categorizing the behavior, we can better understand its nature and take steps toward prevention.
        """
    )
    st.markdown("---")

    # Load model and vectorizer
    model_path = "random_forest_model.joblib"
    vectorizer_path = "vectorizer.joblib"
    model = load_model(model_path)
    vectorizer = load_model(vectorizer_path)

    if model is None or vectorizer is None:
        st.error("Model or vectorizer failed to load. Please check the file paths and try again.")
        return

    # Input Area
    st.markdown("### Enter Your Text Below")
    user_input = st.text_area("Type or paste a sentence:", placeholder="e.g., You're such a loser.")
    detect_button = st.button("🚨 Detect Cyberbullying")

    # Detect
    if detect_button:
        if user_input.strip():
            st.write("🔄 Detecting...")
            # Preprocess the input
            input_data = preprocess_text(user_input, vectorizer)
            
            if input_data is not None:
                # Predict
                prediction = model.predict(input_data)
                category_index = int(prediction[0])

                # Display Results
                st.success(f"🛡️ **Category**: {LABELS[category_index]}")
            else:
                st.error("❌ Error in preprocessing the input text.")
        else:
            st.error("❌ Please enter some text to detect cyberbullying.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        ### Disclaimer
        This tool is for educational purposes only and may not always produce accurate results. 
        Please use it responsibly and seek professional advice when dealing with serious cases of cyberbullying.

        Created with ❤️ by Winaaa.
        """
    )

if __name__ == "__main__":
    main()
