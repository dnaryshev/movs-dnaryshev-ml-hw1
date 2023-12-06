from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

car_model = None
cat_encoder = None
seat_encoder = None
means = None

# Функция для загрузки модели
def load_ridge_model():
    global car_model, cat_encoder, seat_encoder, means
    with open('car_model_ridge.pickle', 'rb') as file:
        data = pickle.load(file)
        car_model = data['model']
        cat_encoder = OneHotEncoder(drop='first', sparse_output=False, categories=data['cat_encoder'])
        seat_encoder = OneHotEncoder(drop='first', sparse_output=False, categories=data['seat_encoder'])
        means = data['means']

# Глобальная переменная для модели
load_ridge_model()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

app = FastAPI()

def preprocess_item(item: Item):
    item_df = pd.DataFrame([item])
    numerical_features = item_df[['year', 'km_driven', 'mileage', 'engine', 'max_power']]
    categorical_features = item_df[['fuel', 'seller_type', 'transmission', 'owner']]

    pass

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    processed_item = preprocess_item(item)
    predicted_price = car_model.predict(processed_item)
    return predicted_price[0]

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    processed_items = [preprocess_item(item) for item in items.objects]
    predicted_prices = car_model.predict(processed_items)
    return predicted_prices.tolist()

