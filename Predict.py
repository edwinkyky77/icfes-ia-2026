import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

# Cargar artefactos
model = load_model("model/modelo_saber11.h5", compile=False)
scaler_X = joblib.load("model/scaler_X.pkl")
scaler_y = joblib.load("model/scaler_y.pkl")

with open("features.json") as f:
    features = json.load(f)

# Función de predicción
def predecir(data_dict):
    
    # Convertir a array
    X = np.array([data_dict[col] for col in features]).reshape(1, -1)
    
    # Escalar
    X_scaled = scaler_X.transform(X)
    
    # Predecir
    y_pred_scaled = model.predict(X_scaled)
    
    # Desescalar
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred[0]