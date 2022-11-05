import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd

path = "/Users/taehoon/Documents/dacon_movies/data/data"
route = path+"/preprocessed_data/preprocessed_"
train, test = pd.read_csv(route+"train.csv").drop("Unnamed: 0",
                                                  axis=1), pd.read_csv(route+"test.csv").drop("Unnamed: 0", axis=1)
params = {
    "n_estimators": 100,
    "learning_rate": 0.04,
    "gamma": 0,
    "subsample": 0.75,
    "colsample_bytree": 1,
    "max_depth": 7,
}
model = xgb.XGBRegressor(**params)
