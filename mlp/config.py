config = {"train_test_size":0.2,
         "source_path": "https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv",
         "output_path": "models/output.pkl",
       "models":{
         "model1":{"algo": "LinearRegression", "params":{"normalize":True}},
         # "model2":{"algo": "LogisticRegression", "params":{"solver":"lbfgs", "multi_class":"auto","n_jobs":-1,"max_iter":2000}},
         # "model3":{"algo": "MLPRegressor", "params":{"solver":"lbfgs","alpha":1e-5,"hidden_layer_sizes":(50, 50, 50, 30, 30),"random_state":1}}
            }
         }