import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10_000
MAX_LEN = 200
EMBED_DIM = 128
RNN_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 8

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
    return (x_train, y_train), (x_test, y_test)


def make_dataset(x, y, batch_size, training=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_simple_rnn():
    model = Sequential(
        [
            Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
            SimpleRNN(RNN_UNITS),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm():
    model = Sequential(
        [
            Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
            LSTM(RNN_UNITS),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_save(model, train_ds, val_ds, save_path):
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )
    model.save(save_path)


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    train_ds = make_dataset(x_train, y_train, BATCH_SIZE, training=True)
    val_ds = make_dataset(x_test, y_test, BATCH_SIZE, training=False)

    rnn = build_simple_rnn()
    rnn_path = os.path.join(MODELS_DIR, "simple_rnn_model.h5")
    train_and_save(rnn, train_ds, val_ds, rnn_path)

    lstm = build_lstm()
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    train_and_save(lstm, train_ds, val_ds, lstm_path)


if __name__ == "__main__":
    main()