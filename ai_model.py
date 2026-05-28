import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber


def build_lstm_model(seq_length=16, num_features=5):
    """
    בונה את רשת הנוירונים העמוקה מסוג LSTM.
    """
    model = Sequential()

    # שכבת LSTM ראשונה עם Dropout למניעת Overfitting (התאמת יתר)
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, num_features)))
    model.add(Dropout(0.2))

    # שכבת LSTM שנייה
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # שכבות Dense סופיות (שכבות פלט)
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # הפלט הסופי: חיזוי מחיר בודד (ה-Close המנורמל הבא)

    # קימפול המודל עם Huber Loss ואופטימייזר Adam
    model.compile(optimizer='adam', loss=Huber())

    return model