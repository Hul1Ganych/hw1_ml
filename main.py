import numpy as np
import pandas as pd
import joblib
import tempfile
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from preproccess import preproccess_total

app = FastAPI()

_model_for_prediction = joblib.load("model.pkl")

df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
subset = [
    'name', 'year', 'km_driven', 'fuel', 'seller_type','transmission', 
    'owner', 'mileage', 'engine', 'max_power', 'torque','seats'
    ]
df_train = df_train.drop_duplicates(keep='first', subset=subset)
df_train.reset_index(drop=True, inplace=True)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[Union[str, float]] 
    engine: Optional[Union[str, float]]
    max_power: Optional[Union[str, float]]
    torque: Optional[Union[str, float]]
    seats: float

    def convert_to_df(self):
        return pd.DataFrame([self.model_dump()])

class Items(BaseModel):
    objects: List[Item]

    @classmethod
    def get_csv(cls, path_to_csv) -> "Items":
        items = pd.read_csv(path_to_csv, dtype=str)
        objects = []
        model_fields = list(Item.model_fields.keys())
        for index, row in items.iterrows():
            objects.append(Item(**row[model_fields].to_dict()))
        return Items(objects=objects)
    
    def convert_to_df(self):
        return pd.DataFrame([item.model_dump() for item in self.objects])

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df_train_copy = df_train.copy()
    data = item.convert_to_df()
    data = preproccess_total(data_train=df_train_copy, data_test=data)

    return np.nan_to_num(_model_for_prediction.predict(data)[0])


@app.post("/predict_items")
def predict_items(file_csv: UploadFile) -> FileResponse:
    df_train_copy = df_train.copy()
    items = Items.get_csv(file_csv.file)
    data = items.convert_to_df()
    data = preproccess_total(data_train=df_train_copy, data_test=data)

    data["predict"] = np.nan_to_num(_model_for_prediction.predict(data))

    with tempfile.NamedTemporaryFile() as temp:
        data.to_csv(f"{temp.name}.csv")
        return FileResponse(f"{temp.name}.csv")