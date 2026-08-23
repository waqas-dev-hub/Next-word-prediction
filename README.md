
Author-Abu Waqas

---


# Next Word Prediction using LSTM (FastAPI + Streamlit)

A Deep Learning application that predicts the next word in a sequence using an LSTM model trained on Shakespeare's *Hamlet*. Built with a FastAPI backend and a Streamlit frontend.

---

## Project Architecture

- Backend.py: FastAPI backend that loads the trained LSTM model and tokenizer, exposing a /predict REST endpoint.
- Frontend.py: Streamlit interactive UI that communicates with the FastAPI backend.
- app.py: Standalone Streamlit application (monolithic version).
- next_word_lstm.h5: Trained LSTM model weights.
- Tokenizer.pickle: Keras Tokenizer mapping words to sequence IDs.
- experiemnts.ipynb: Training and experimentation notebook.

---
