from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import pandas as pd
import os
import joblib
from onedrivedownloader import download

# Инициализация приложения и модели
app = FastAPI()

URL = "https://niuitmo-my.sharepoint.com/:u:/g/personal/askernagoev_niuitmo_ru/IQAUA1jAY9rVSYaVVfOKKsorAQ57yrpMDeCNepKY_fQHIXo?e=JYI7d9"
if not os.path.isfile("rf_model.pkl"):
    download(URL, "rf_model.pkl", unzip=False)

with open ("rf_model.pkl", 'rb') as file:
    model = joblib.load(file)

class ModelRequestData(BaseModel):
    total_square: float
    rooms: int
    floor: int

class Result(BaseModel):
    result: int

# Получение предсказания
input_df = pd.DataFrame({
    "total_square": 75,
    "rooms": 2,
    "floor": 5
}, index =[0])

price_predicted = int(model.predict(input_df)[0])

# Реализация запросов
@app.get("/health")
def health():
    return JSONResponse(content={"message": "Сервер работает"}, status_code=200)

@app.get("/predict_get")
def predict_get():
    return JSONResponse(content={"message": "Предсказанный результат: " + str(price_predicted)})

@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.model_dump()
    input_df = pd.DataFrame(input_data, index=[0])
    result = int(model.predict(input_df)[0])
    return Result(result=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)