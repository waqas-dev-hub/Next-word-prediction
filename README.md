# Next Word Prediction using LSTM (FastAPI + Streamlit)

A Deep Learning application that predicts the next word in a sequence using an LSTM model trained on Shakespeare's *Hamlet*. Built with a FastAPI backend and a Streamlit frontend.

---

## Project Architecture

- **ackend.py**: FastAPI backend that loads the trained LSTM model and tokenizer, exposing a /predict REST endpoint.
- **rontend.py**: Streamlit interactive UI that communicates with the FastAPI backend.
- **pp.py**: Standalone Streamlit application (monolithic version).
- **
ext_word_lstm.h5**: Trained LSTM model weights.
- **	okenizer.pickle**: Keras Tokenizer mapping words to sequence IDs.
- **experiemnts.ipynb**: Training and experimentation notebook.

---

## Installation & Setup

1. **Clone the repository:**
   \\ash
   git clone <YOUR_REPOSITORY_URL>
   cd <REPO_FOLDER>
   \
2. **Create and activate a virtual environment:**
   \\ash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   \
3. **Install dependencies:**
   \\ash
   pip install -r requirements.txt
   \
---

## Running the Application

### Option A: FastAPI Backend + Streamlit Frontend (Recommended)

1. **Start the FastAPI Backend:**
   \\ash
   uvicorn backend:app --reload
   \   - API Docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health check: [http://localhost:8000/health](http://localhost:8000/health)

2. **Start the Streamlit Frontend (in a new terminal):**
   \\ash
   streamlit run frontend.py
   \   - Streamlit Web App: [http://localhost:8501](http://localhost:8501)

### Option B: Standalone Streamlit App

\\ash
streamlit run app.py
\