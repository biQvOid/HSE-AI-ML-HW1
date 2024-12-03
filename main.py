from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi import File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import json
import io
import pickle
from fastapi.responses import StreamingResponse

app = FastAPI()

feats = [
    "name", "year", "km_driven", "fuel", "seller_type", "transmission",
    "owner", "mileage", "engine", "max_power", "seats"
]

cat_feats = ["brand", "seats", "fuel", "seller_type", "transmission", "owner"]
num_feats = ["year", "km_driven", "mileage", "engine", "max_power"]

def split_values(str_value):
    if str_value is np.nan:
        return np.nan
    num_value = str_value.split()[0]
    try:
        num_value = float(num_value)
    except:
        num_value = 0.0
    return num_value

def preprocess(data):
    data["mileage"] = data["mileage"].apply(lambda x: split_values(x))
    data["engine"] = data["engine"].apply(lambda x: split_values(x))
    data["max_power"] = data["max_power"].apply(lambda x: split_values(x))
    return data

def get_brand(data):
    data["brand"] = data["name"].apply(lambda x: x.split()[0])
    data.drop(columns=["name"], inplace=True)
    return data

def pipeline(data):
    data = get_brand(data)
    with open('scaler.pickle','rb') as file:
        scaler = pickle.load(file)
    with open('encoder.pickle','rb') as file:
        encoder = pickle.load(file)
    with open('model.pickle','rb') as file:
        model = pickle.load(file)
    scaled_feats = scaler.transform(data[num_feats])
    encoded_feats = encoder.transform(data[cat_feats])
    feats_ = np.concatenate([scaled_feats, encoded_feats], axis=1)
    predictions = model.predict(feats_)
    return predictions

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: float

class Item2(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item2) -> float:
    item_ = Item2.model_validate(item)
    df = pd.DataFrame.from_dict([item_.model_dump()])
    with open('fill_values.json') as json_file:
        fill_values = json.load(json_file)
    data = preprocess(df)
    data = data.fillna(fill_values)
    prediction = pipeline(data)
    return prediction


@app.get("/")
def home_page():
    return {"message": "Привет, Хабр!"}


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
    df = pd.read_csv(file.filename)
    with open('fill_values.json') as json_file:
        fill_values = json.load(json_file)
    test_data = df.copy()
    test_data = preprocess(test_data)
    test_data = test_data.fillna(fill_values)
    df_records = test_data[feats].to_dict("records")
    df_items = [Item.model_validate(record) for record in df_records]
    items = Items.model_validate({"objects": df_items})
    predictions = pipeline(test_data)
    df["price_prediction"] = pd.Series(predictions)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    
    return response
