import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class _KerasUnpickler(pickle.Unpickler):
    """Remap tokenizers pickled under Keras 3 paths to tf_keras (Keras 2)."""

    def find_class(self, module: str, name: str):
        if module.startswith("keras.src."):
            module = module.replace("keras.src.", "tf_keras.", 1)
        elif module.startswith("keras.") and not module.startswith("keras_"):
            module = module.replace("keras.", "tf_keras.", 1)
        return super().find_class(module, name)


model = load_model("next_word_lstm.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = _KerasUnpickler(handle).load()

MAX_SEQUENCE_LEN = model.input_shape[1] + 1


app = FastAPI(title="Next Word Prediction API")


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    input_text: str
    next_word: str | None


def predict_next_word(text: str) -> str | None:
    """Predict the next word given an input text sequence."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= MAX_SEQUENCE_LEN:
        token_list = token_list[-(MAX_SEQUENCE_LEN - 1):]
    token_list = pad_sequences(
        [token_list], maxlen=MAX_SEQUENCE_LEN - 1, padding="pre"
    )
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    next_word = predict_next_word(request.text)
    return PredictionResponse(input_text=request.text, next_word=next_word)
