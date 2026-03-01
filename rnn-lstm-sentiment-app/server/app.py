import os
import re
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 200

# IMDB dataset uses special reserved indices:
# 0 = <PAD>, 1 = <START>, 2 = <OOV>, 3+ = actual words.
INDEX_FROM = 3
OOV_INDEX = 2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

rnn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "simple_rnn_model.h5"))
lstm_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))

_raw_word_index = imdb.get_word_index()
word_index = {word: (idx + INDEX_FROM) for word, idx in _raw_word_index.items()}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


def encode_text(text):
    cleaned = re.sub(r"[^a-zA-Z0-9']", " ", text.lower())
    tokens = cleaned.split()
    # Include the <START> token so encoding matches training data.
    encoded = [1]
    for word in tokens:
        idx = word_index.get(word, OOV_INDEX)
        if idx < VOCAB_SIZE:
            encoded.append(idx)
    padded = pad_sequences([encoded], maxlen=MAX_LEN, padding="post", truncating="post")
    return padded


def predict(model, text):
    encoded = encode_text(text)
    score = float(model.predict(encoded, verbose=0)[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    return {"score": round(score, 4), "label": label}


@app.post("/predict")
def predict_sentiment(input: TextInput):
    rnn_result = predict(rnn_model, input.text)
    lstm_result = predict(lstm_model, input.text)
    return {
        "rnn_result": rnn_result,
        "lstm_result": lstm_result,
    }