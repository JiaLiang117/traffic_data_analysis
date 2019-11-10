import pickle
from math import sqrt
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error

import sys


def main(model_path, data_path, model_choice):

    with open(model_path,"rb") as model:
        models = pickle.load(model)

    data = pd.read_csv(data_path)
    data = standardise_data(data, models)

    x_columns = [x for x in list(data.columns) if x != "traffic_volume"]
    x = data[x_columns]

    x = models["transform"]["standard_scalar"].transform(x)
    y = data["traffic_volume"]

    model = models[model_choice]["model"]
    y_pred = model.predict(x)

    rms = calculate_rms(y, y_pred)
    print(rms)

    return y_pred


def calculate_rms(actual, predict):
    return sqrt(mean_squared_error(actual, predict))


def standardise_data(data, models):

    data["date_time"] = pd.to_datetime(data["date_time"])
    data["date"] = data["date_time"].apply(lambda x: datetime.strftime(x,"%Y-%m-%d"))
    data["time"] = data["date_time"].apply(lambda x: datetime.strftime(x,"%H:%M"))
    data["holiday"] = data["holiday"].apply(lambda x : 1 if x != "None" else 0)
    
    weather = pd.get_dummies(data["weather_main"])
    data = pd.concat([data, weather],axis=1)
    
    data["time"] = data["time"].apply(lambda x: int(x[:2]))
    time = pd.get_dummies(data["time"])
    data = pd.concat([data, time],axis=1)

    data.drop(
        ["weather_main","weather_description","date_time","date","time"],
        axis=1,
        inplace=True)

    data_columns = models["transform"]["data_columns"]
    scaler = models["transform"]["standard_scalar"]

    columns_to_drop = [col for col in list(data.columns) if col not in data_columns]
    data.drop(columns_to_drop, axis=1, inplace=True)

    columns_to_add = [col for col in data_columns if col not in list(data.columns)]

    for col in columns_to_add:
        data[col] = 0

    data = data[data_columns]
    
    return data


if __name__ == "__main__":

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    model_choice = sys.argv[3]
    main(model_path, data_path, model_choice)