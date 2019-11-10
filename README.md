configure pipline using mlp/config.py  
train_test_size : how much % of your data will be allocate to test set, float(0,1)  
source_path: training data source path  
output_path: pickled model output path  
models: configuration models that you wish to train in the pipeline,  
    - algo: model to be used, currently only supporting (LinearRegression, LogisticRegression, MLPRegressor)  
    - params: dictionary of configurable params for the models  
  
how to run?  
bash run.sh will start the script  
User input will be required on whether you want to re-train the model  
After traning/not training,  
User input will be required on which model to deploy, a csv file with the prediction will be generated at the output path  
  
mlp/deploy.py takes 3 arguments  
location of model pickle file  
location of test date file  
path of location to write the output predictions  