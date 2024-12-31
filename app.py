import streamlit as st
import numpy as np
import tensorflow as tf

# Load the TFLite model
def load_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess text (You may need to adapt this based on your model's input requirements)
def preprocess_text(text, max_length=100):
    # Example preprocessing: Truncate or pad text to a fixed length
    # Add any tokenizer or transformation specific to your training process
    # This is just a placeholder.
    return np.zeros((1, max_length), dtype=np.float32)

# Perform prediction
def predict(interpreter, input_data):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    return prediction

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
    st.image("https://via.placeholder.com/800x300.png?text=Cyberbullying+Awareness", use_column_width=True)
    st.markdown("---")

    # Description Section
    st.markdown(
        """
        ### How It Works
        1. Enter any text in the input box below. 
        2. Click on the **Detect Cyberbullying** button. 
        3. The app will classify the text into one of the following categories:
           - **Religion**
           - **Age**
           - **Gender**
           - **Ethnicity**
           - **Not Bullying**

        ### Why This Matters
        Identifying cyberbullying is crucial in combating harmful online interactions. 
        By categorizing the behavior, we can better understand its nature and take steps toward prevention.
        """
    )
    st.markdown("---")

    # Load model
    model_path = "random_forest_model.tflite"  # Update the path if needed
    interpreter = load_model(model_path)

    # Input Area
    st.markdown("### Enter Your Text Below")
    user_input = st.text_area("Type or paste a sentence:", placeholder="e.g., You're such a loser.")
    detect_button = st.button("🚨 Detect Cyberbullying")

    # Detect
    if detect_button:
        if user_input.strip():
            st.write("🔄 Detecting...")
            # Preprocess the input
            input_data = preprocess_text(user_input)
            
            # Predict
            prediction = predict(interpreter, input_data)
            category_index = np.argmax(prediction)
            confidence = np.max(prediction)

            # Display Results
            st.success(f"🛡️ **Category**: {LABELS[category_index]}")
            st.info(f"🔢 **Confidence Score**: {confidence:.2f}")
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
