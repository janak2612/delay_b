
import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
from groq import Groq

# --- Page Configuration ---
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")
st.title("🚚 Delivery Delay Predictor")
st.markdown("Enter delivery conditions in natural language to predict potential delays.")

# --- Load Model and Feature Names ---
# Load the trained logistic regression model
try:
    logi = joblib.load("logi.sav")
except FileNotFoundError:
    st.error("Error: 'logi.sav' model file not found. Please ensure the model is trained and saved.")
    st.stop()

# Define feature names (from X.columns in the notebook)
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# --- Groq API Key Input ---
groq_api_key = st.text_input("Enter your Groq API Key:", type="password", help="You can get your API key from app.groq.com")

# --- predict_with_natural_language Function ---
def predict_with_natural_language(natural_language_input, model, feature_names, groq_client):
    """
    Uses the Groq API to parse natural language input and predict with the given model.

    Args:
        natural_language_input (str): The natural language description of the delivery conditions.
        model: The trained scikit-learn model (e.g., LogisticRegression).
        feature_names (list): A list of feature names that the model expects.
        groq_client: An initialized Groq client object.

    Returns:
        tuple: (prediction, probabilities, extracted_features) or (None, None, None) on error.
    """

    prompt = f"""You are an expert in parsing delivery condition descriptions. \nExtract the following {len(feature_names)} numerical features from the given natural language input. \nPresent them as a Python list of integers or floats, in the exact order specified. \nIf a value is not explicitly mentioned, try to infer a reasonable default or return 'None' for it. \nDo not include any other text or explanation, just the list. \nMake sure the list has exactly {len(feature_names)} elements.\n\nFeatures in order: {feature_names}\n\nNatural language input: {natural_language_input}"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b", # The specified Groq model
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            stream=False
        )

        groq_output = completion.choices[0].message.content
        st.sidebar.text(f"Groq output: {groq_output}") # Display Groq output in sidebar for debugging

        extracted_features = eval(groq_output)
        if not isinstance(extracted_features, list):
            raise ValueError("Groq did not return a list.")
        if len(extracted_features) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, but Groq returned {len(extracted_features)}.")
        
        # Handle None values by converting to a default (e.g., 0 or mean of column if applicable)
        # For this example, let's convert None to 0, but in a real scenario, more sophisticated imputation might be needed.
        extracted_features_processed = [0 if x is None else x for x in extracted_features]

        input_data = np.array([extracted_features_processed], dtype=float)
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        return prediction, probabilities, extracted_features_processed
    except Exception as e:
        st.error(f"Error processing request: {e}")
        return None, None, None

# --- Natural Language Input ---
user_input = st.text_area("Describe the delivery conditions here:",
                          "A delivery of 15 miles, moderate traffic, clear weather, afternoon slot, experienced driver (10 years), 5 stops, new vehicle (2 years old), good road conditions (score 4), package weight 5kg, good fuel efficiency (15 km/l), and 60 minutes processing time.",
                          height=150)

# --- Prediction Button ---
if st.button("Predict Delay"):
    if not groq_api_key:
        st.warning("Please enter your Groq API Key to proceed.")
    elif not user_input:
        st.warning("Please enter a natural language description.")
    else:
        try:
            client = Groq(api_key=groq_api_key)
            prediction, probabilities, features = predict_with_natural_language(
                user_input, logi, feature_names, client
            )

            if prediction is not None:
                st.subheader("Prediction Results:")
                st.write(f"**Natural Language Input:** {user_input}")
                st.write(f"**Extracted Features:** {features}")
                st.write(f"**Prediction:** {'⚠️ Delay Expected' if prediction == 1 else '✅ No Delay Expected'}")
                st.write(f"**Probability of No Delay:** {probabilities[0]:.2f}")
                st.write(f"**Probability of Delay:** {probabilities[1]:.2f}")
            else:
                st.error("Could not make a prediction based on the natural language input. Please check your input and API key.")
        except Exception as e:
            st.error(f"Failed to initialize Groq client or make API call: {e}")
            st.info("Please ensure your Groq API key is valid and has the necessary permissions.")
