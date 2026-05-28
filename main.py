from fastapi import FastAPI, HTTPException
import requests
import mysql.connector
import pandas as pd
from datetime import datetime
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn # הוספנו את uvicorn כדי להריץ את השרת בענן

# ייבוא הפונקציות המקומיות שלנו
from data_processor import prepare_lstm_data
from ai_model import build_lstm_model

app = FastAPI(title="Smart Trade AI Service")

# הגדרות החיבור למסד הנתונים
DB_CONFIG = {
    "host": "mysql-183c5b55-smart-trade-db.j.aivencloud.com",
    "port": 27746,
    "user": "avnadmin",
    "password": os.environ.get("DB_PASSWORD", ""), # <--- התיקון הקריטי!
    "database": "defaultdb",
    "ssl_ca": ""
}

@app.get("/")
def home():
    return {"status": "AI Service is running"}


# ==========================================
# פונקציה 1: עדכון מחיר נוכחי ב-MySQL
# ==========================================
@app.post("/api/ai/update-price/{symbol}")
def update_asset_price(symbol: str):
    binance_symbol = f"{symbol.upper()}USDT"
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found on Binance.")

        data = response.json()
        current_price = float(data["price"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from Binance: {str(e)}")

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            UPDATE assets 
            SET current_price = %s, last_updated = %s 
            WHERE symbol = %s
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(query, (current_price, now, symbol.upper()))
        conn.commit()

        rows_affected = cursor.rowcount
        cursor.close()
        conn.close()

        if rows_affected == 0:
            return {"message": f"Asset {symbol} updated to {current_price}, but symbol wasn't found in DB."}

        return {
            "symbol": symbol.upper(),
            "status": "Updated successfully in DB",
            "current_price": current_price,
            "last_updated": now
        }
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {str(err)}")


# ==========================================
# פונקציה 2: משיכת נרות היסטוריים ועיבודם
# ==========================================
@app.get("/api/ai/candles/{symbol}")
def get_historical_candles(symbol: str, timeframe: str = "4h", limit: int = 64):
    binance_symbol = f"{symbol.upper()}USDT"
    url = f"https://api.binance.com/api/v3/klines"

    params = {
        "symbol": binance_symbol,
        "interval": timeframe,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=400,
                                detail=f"Failed to fetch candles. Binance responded with {response.status_code}")

        raw_candles = response.json()

        df = pd.DataFrame(raw_candles, columns=[
            "Open_Time", "Open", "High", "Low", "Close", "Volume",
            "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
            "Taker_Buy_Base_Asset_Volume", "Taker_Buy_Quote_Asset_Volume", "Ignore"
        ])

        df = df[["Open_Time", "Open", "High", "Low", "Close", "Volume"]].copy()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)

        df["Open_Time"] = pd.to_datetime(df["Open_Time"], unit='ms')
        processed_data = df.to_dict(orient="records")

        # בדיקת מערכת הנרמול שלנו
        X, y, scaler = prepare_lstm_data(processed_data, seq_length=16)

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "total_candles": len(processed_data),
            "X_shape": X.shape,
            "y_shape": y.shape,
            "sample_normalized_candle": X[0][0].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing candles data: {str(e)}")


# ==========================================
# פונקציה 3: אימון מודל ה-LSTM (Deep Learning)
# ==========================================
@app.post("/api/ai/train/{symbol}")
def train_model(symbol: str, timeframe: str = "4h"):
    binance_symbol = f"{symbol.upper()}USDT"
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": timeframe, "limit": 1000}

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch training data.")

        raw_candles = response.json()
        df = pd.DataFrame(raw_candles, columns=[
            "Open_Time", "Open", "High", "Low", "Close", "Volume",
            "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
            "Taker_Buy_Base_Asset_Volume", "Taker_Buy_Quote_Asset_Volume", "Ignore"
        ])
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        processed_data = df.to_dict(orient="records")

        X, y, scaler = prepare_lstm_data(processed_data, seq_length=16)
        model = build_lstm_model(seq_length=16, num_features=5)

        # אימון
        history = model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, verbose=0)

        # שמירה
        model_filename = f"{symbol.lower()}_lstm_model.h5"
        model.save(model_filename)

        final_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        return {
            "symbol": symbol.upper(),
            "status": "Training completed successfully",
            "model_saved_as": model_filename,
            "data_points_trained": len(X),
            "final_training_loss": float(final_loss),
            "final_validation_loss": float(val_loss)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


# ==========================================
# פונקציה 4: חיזוי מחיר עתידי (Inference)
# ==========================================
@app.get("/api/ai/predict/{symbol}")
def predict_next_price(symbol: str, timeframe: str = "4h"):
    model_filename = f"{symbol.lower()}_lstm_model.h5"

    if not os.path.exists(model_filename):
        raise HTTPException(status_code=404, detail=f"Model for {symbol} not found. Please train first.")

    try:
        # 1. טעינת המודל המאומן
        model = load_model(model_filename)

        # 2. משיכת נתונים מהבורסה
        binance_symbol = f"{symbol.upper()}USDT"
        url = f"https://api.binance.com/api/v3/klines"
        params = {"symbol": binance_symbol, "interval": timeframe, "limit": 100}

        response = requests.get(url, params=params)
        raw_candles = response.json()

        df = pd.DataFrame(raw_candles, columns=[
            "Open_Time", "Open", "High", "Low", "Close", "Volume",
            "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
            "Taker_Buy_Base_Asset_Volume", "Taker_Buy_Quote_Asset_Volume", "Ignore"
        ])
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # 3. נרמול הנתונים בדיוק כמו באימון
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        # 4. הכנת הקלט לחיזוי (16 הנרות האחרונים)
        last_16_candles = scaled_data[-16:]
        X_predict = np.array([last_16_candles])

        # 5. חיזוי המודל
        predicted_scaled_close = model.predict(X_predict)

        # 6. המרה חזרה למחיר בדולרים
        dummy_row = np.zeros((1, 5))
        dummy_row[0, 3] = predicted_scaled_close[0][0]
        predicted_real_price = scaler.inverse_transform(dummy_row)[0, 3]

        # 7. קבלת החלטה ותיעוד הזמן
        current_price = df.iloc[-1]["Close"]
        trend = "UP" if predicted_real_price > current_price else "DOWN"
        action = "BUY" if trend == "UP" else "SELL"

        # שולפים את התאריך והשעה של הרגע הזה בדיוק
        prediction_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return {
            "symbol": symbol.upper(),
            "timestamp": prediction_time,
            "current_price": float(current_price),
            "predicted_next_close": float(round(predicted_real_price, 2)),
            "predicted_trend": trend,
            "recommended_action": action
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ==========================================
# הרצת השרת (קריטי עבור פריסה בענן כמו Render)
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)