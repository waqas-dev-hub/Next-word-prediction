import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("Next Word Prediction With LSTM And Early Stopping")

input_text = st.text_input("Enter the sequence of Words", "To be or not to")

if st.button("Predict Next Word"):
    with st.spinner("Predicting..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict",
                json={"text": input_text},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            st.success(f"Next word: **{data['next_word']}**")
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the backend. "
                "Make sure the FastAPI server is running on "
                f"{BACKEND_URL}"
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
