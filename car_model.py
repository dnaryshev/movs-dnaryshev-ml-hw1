from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import math
import uvicorn
import sklearn

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
        cat_encoder = data['cat_encoder']
        seat_encoder = data['seat_encoder']
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


def to_float(s):
    if isinstance(s, float) and math.isnan(s):
        return pd.NA
    try:
        return float(s.strip().split()[0])
    except ValueError:
        return pd.NA


def preprocess_item(item: Item):
    item_df = pd.DataFrame([item.model_dump()])
    item_df.drop(['torque', 'selling_price', 'name'], axis=1, inplace=True)
    numerical_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    for column in ['mileage', 'engine', 'max_power']:
        item_df[column] = item_df[column].apply(to_float)
    item_df[numerical_features] = item_df[numerical_features].fillna(means)
    categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
    item_cat_encoded = cat_encoder.transform(item_df[categorical_features])
    item_cat_encoded_df = pd.DataFrame(item_cat_encoded,
                                       columns=cat_encoder.get_feature_names_out(categorical_features))
    seats_cat_encoded = seat_encoder.transform(item_df[['seats']])
    seats_cat_encoded_df = pd.DataFrame(seats_cat_encoded, columns=seat_encoder.get_feature_names_out(['seats']))
    item_df.drop(columns=categorical_features + ['seats'], inplace=True)
    x_item = pd.concat([item_df, item_cat_encoded_df, seats_cat_encoded_df], axis=1)
    return x_item


# Значение для тестирования
item_data = Item(
    name='Maruti Swift Dzire VDI',
    year=2014,
    selling_price=450000,
    km_driven=145500,
    fuel='Diesel',
    seller_type='Individual',
    transmission='Manual',
    owner='First Owner',
    mileage='23.4 kmpl',
    engine='1248 CC',
    max_power='74 bhp',
    torque='190Nm@ 2000rpm',
    seats=5.0
)


app = FastAPI()


@app.get("/health")
def get_root():
    return "ML dnaryshev model #HW1"


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    processed_item = preprocess_item(item)
    predicted_price = car_model.predict(processed_item)
    return predicted_price[0]


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    processed_items = [preprocess_item(item) for item in items.objects]
    processed_items_df = pd.concat(processed_items, ignore_index=True)
    predicted_prices = car_model.predict(processed_items_df)
    return predicted_prices.tolist()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)