import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_lstm_data(candles_list, seq_length=16):
    """
    מקבלת רשימת נרות, מנרמלת אותם לטווח 0-1,
    ומייצרת רצפים (X) ותוויות (Y) לאימון/חיזוי במודל LSTM.
    """
    # 1. הפיכת רשימת המילונים ל-DataFrame של Pandas
    df = pd.DataFrame(candles_list)

    # 2. חילוץ רק העמודות הרלוונטיות למודל (OHLCV)
    features_df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # 3. נרמול הנתונים לטווח של 0 עד 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features_df)

    # 4. חלוקה לרצפי זמן (Sliding Window)
    X = []
    y = []

    # עוברים על הנתונים ויוצרים "חלונות" באורך seq_length
    for i in range(len(scaled_data) - seq_length):
        # הקלט (X) הוא רצף של נרות (למשל 16 נרות)
        X.append(scaled_data[i:i + seq_length])

        # הפלט (Y) הוא מחיר הסגירה (Close) של הנר הבא אחרי הרצף.
        # אינדקס 3 מייצג את עמודת ה-Close במערך ה-OHLCV שלנו.
        y.append(scaled_data[i + seq_length, 3])

    return np.array(X), np.array(y), scaler