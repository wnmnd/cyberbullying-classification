import streamlit as st
import pickle
import numpy as np

# Load the Random Forest model
def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Preprocess the input text
def preprocess_text(text, max_length=100):
    # Convert text to lowercase and truncate/pad to a fixed length
    processed = text.lower()
    return processed

# Streamlit App
def main():
    st.set_page_config(page_title="Cyberbullying Detection", page_icon="🔍", layout="wide")

    # Header
    st.title("🔍 Cyberbullying Detection App")
    st.markdown(
        """
        Welcome to the **Cyberbullying Detection App**! This tool uses a Random Forest model 
        to analyze text and classify it into one of five categories.
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
        """
    )
    st.markdown("---")

    # Load model
    model_path = "random_forest_model.pkl"  # Update this if the file is in a different location
    model = load_model(model_path)

    # Input Area
    st.markdown("### Enter Your Text Below")
    user_input = st.text_area("Type or paste a sentence:", placeholder="e.g., You're such a loser.")
    detect_button = st.button("🚨 Detect Cyberbullying")

    # Detect
    if detect_button:
        if user_input.strip():
            st.write("🔄 Detecting...")
            # Preprocess the input
            preprocessed_text = preprocess_text(user_input)
            
            # Vectorize the text (Assuming a vectorizer was used during model training)
            try:
                with open("vectorizer.pkl", "rb") as vec_file:
                    vectorizer = pickle.load(vec_file)
                input_data = vectorizer.transform([preprocessed_text])
                
                # Predict
                prediction = model.predict(input_data)
                confidence = np.max(model.predict_proba(input_data))

                # Map prediction to category labels
                LABELS = [
                    'Religion-based Cyberbullying',
                    'Age-based Cyberbullying',
                    'Gender-based Cyberbullying',
                    'Ethnicity-based Cyberbullying',
                    'Not Bullying'
                ]
                category = LABELS[prediction[0]]

                # Display Results
                st.success(f"🛡️ **Category**: {category}")
                st.info(f"🔢 **Confidence Score**: {confidence:.2f}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.error("❌ Please enter some text to detect cyberbullying.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        ### Disclaimer
        This tool is for educational purposes only and may not always produce accurate results. 
        Please use it responsibly and seek professional advice when dealing with serious cases of cyberbullying.

        Created with ❤️ by [Your Name].
        """
    )

if __name__ == "__main__":
    main()
