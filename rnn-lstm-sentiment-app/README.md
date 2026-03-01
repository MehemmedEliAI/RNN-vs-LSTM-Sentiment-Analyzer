## RNN vs LSTM Sentiment Analyzer

This project compares a simple RNN and an LSTM for movie-review sentiment analysis on the IMDB dataset. The backend is built with FastAPI and TensorFlow/Keras, and the frontend is a small HTML/CSS/JS app that visualizes both models' predictions.

### Project structure

- **training/** – scripts to train and save the RNN and LSTM models.
- **server/** – FastAPI app that loads the trained models and exposes a `/predict` endpoint.
- **frontend/** – static UI that sends text to the backend and visualizes the two models' scores.

### Setup

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate  # on Windows
   ```

2. Install backend dependencies:

   ```bash
   pip install -r server/requirements.txt
   ```

### Train the models

The training script downloads the IMDB dataset, builds both models, trains them, and saves them into `models/`.

```bash
python training/train_models.py
```

By default, it trains for 8 epochs. You can change `EPOCHS` in `training/train_models.py` if you want faster training or better accuracy.

### Run the API server

From the project root:

```bash
uvicorn server.app:app --reload
```

The sentiment endpoint will be available at:

- `POST http://127.0.0.1:8000/predict`

Request body:

```json
{ "text": "your movie review here" }
```

### Run the frontend

Open `frontend/index.html` in a browser (or serve the `frontend/` folder with any static-file server). Make sure the FastAPI server is running, then:

1. Type a movie review in the text area.
2. Click **Analyze sentiment**.
3. The UI will show each model's score, a color-coded prediction pill (positive/negative), and an animated confidence bar per model.

### Notes

- If the predictions look too similar or not very accurate, try increasing `EPOCHS` and retraining.
- The backend includes basic text cleaning and uses the same IMDB word index as used during training so that inference matches training conditions.

