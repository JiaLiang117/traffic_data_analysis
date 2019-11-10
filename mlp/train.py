import pickle
import json
from math import sqrt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


from config import config



class Train(object):

    def __init__(self, source_path, config):
        self.source_path = source_path # path
        self.train_test_size = config["train_test_size"] #json
        self.standard_scalar = None
        self.data_columns = None
        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None

    def extract(self):
        """
        method to extract, clean and prepare training data
        returns x_train, x_valid, y_train, y_valid 
        """
        data = pd.read_csv(self.source_path)
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

        self.data_columns = list(data.columns)

        x_columns = [x for x in list(data.columns) if x != "traffic_volume"]
        x = data[x_columns]
        y = data["traffic_volume"]

        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.train_test_size)
        
        scaler = StandardScaler()  
        scaler.fit(x_train) 
        self.standard_scalar = scaler
        x_train = scaler.transform(x_train)  
        x_valid = scaler.transform(x_valid)
        
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid


    def models(self, key, config):
        model = globals()[config["algo"]](**config["params"])
        model.fit(self.x_train, self.y_train)                         
        y_train_pred = model.predict(self.x_train)
        y_valid_pred = model.predict(self.x_valid)
        rms_train = calculate_rms(self.y_train, y_train_pred)
        rms_valid = calculate_rms(self.y_valid, y_valid_pred)

        print("for: {}".format(key))
        print("base_rms_train: {}".format(rms_train))
        print("base_rms_valid: {}".format(rms_valid))

        return {"model": model,
        "rms_train": rms_train,
        "rms_valid": rms_valid}


class BaseCase(object):
    def __init__(self, train):
        self.mean = train.y_train.mean()
        y_train_pred = np.full((train.y_train.shape[0],), self.mean)
        y_valid_pred = np.full((train.y_valid.shape[0],), self.mean)
        self.base_rms_train = calculate_rms(train.y_train, y_train_pred)
        self.base_rms_valid = calculate_rms(train.y_valid, y_valid_pred)
        print("for: baseCase")
        print("base_rms_train: {}".format(self.base_rms_train))
        print("base_rms_valid: {}".format(self.base_rms_valid))

    def predict(self, x_train):
        return np.full((x_train.shape[0],), self.mean)




def calculate_rms(actual, predict):
    return sqrt(mean_squared_error(actual, predict))



def main():

    
    source_path = config["source_path"]
    output_path = config["output_path"]

    train = Train(source_path,config)
    train.extract()

    output = {}

    # calculate base case
    baseCase = BaseCase(train)

    output["baseCase"] = {
        "baseCase":
        {"model": baseCase.mean,
        "rms_train": baseCase.base_rms_train,
        "rms_valid": baseCase.base_rms_valid}
    }


    models = config["models"]

    for key in models:
        model = train.models(key, models[key])
        output[key] = model   

    output["transform"] = {}
    output["transform"]["standard_scalar"] = train.standard_scalar
    output["transform"]["data_columns"] = train.data_columns

    with open(output_path, "wb") as output_path:
        pickle.dump(output, output_path)




if __name__ == "__main__":
    main()