import xgboost as xgb
import catboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from bayes_opt import BayesianOptimization
import numpy as np
import time

path = "/Users/taehoon/Documents/DACON/example_movies/dacon_movies/data"
route = path+"/preprocessed_data/preprocessed_"
train, test = pd.read_csv(
    route+"train.csv", index_col=0), pd.read_csv(route+"test.csv", index_col=0)
drops = ["genre", "screening_rat", "release_day"]
train = train.drop(drops, axis=1)
test = test.drop(drops, axis=1)


def label_encoding(train, test):
    all_df = pd.concat([train, test], ignore_index=True)
    label_encoder = {}
    str_columns = [(i, j) for i, j in zip(train.dtypes, train.columns)]
    for tup in str_columns:
        dtype, column = tup
        if dtype != "object":
            continue
        val2idx = {i: j for j, i in enumerate(all_df[column].unique())}
        label_encoder[column] = val2idx
        train[column] = train[column].map(val2idx)
        test[column] = test[column].map(val2idx)
    return train, test, label_encoder


train, test, _ = label_encoding(train, test)

train_x, test_x, train_y, test_y = train_test_split(
    train.drop("box_off_num", axis=1), train["box_off_num"])


def cbt_reg(n_estimators, depth, learning_rate, subsample, l2_leaf_reg):
    params = {
        "n_estimators": int(n_estimators),
        "depth": int(depth),
        "learning_rate": learning_rate,
        "subsample": subsample,
        "l2_leaf_reg": l2_leaf_reg,
    }
    cbtr_model = catboost.CatBoostRegressor(
        **params,
        bootstrap_type='Bernoulli',
        eval_metric='RMSE',
        od_type='Iter',
        allow_writing_files=False)
    cbtr_model.fit(train_x, train_y, silent=True)
    y_pred = cbtr_model.predict(test_x)
    rmse = mean_squared_error(test_y, y_pred, squared=False)
    r2 = r2_score(test_y, y_pred)
    return 1-rmse


pbounds = {"n_estimators": (500, 1000),
           "depth": (2, 7),
           "learning_rate": (.01, 0.2),
           "subsample": (0.6, 1.),
           "l2_leaf_reg": (0, 10),
           }
bo = BayesianOptimization(f=cbt_reg, pbounds=pbounds,
                          verbose=2, random_state=42)
bo.maximize(init_points=2, n_iter=500, acq='ei', xi=0.01)
high_score = bo.max
1-cbt_reg(**high_score["params"])


for param in ('depth', "n_estimators"):
    high_score["params"][param] = int(high_score["params"][param])
final_model = catboost.CatBoostRegressor(
    **high_score["params"],
    bootstrap_type='Bernoulli',
    eval_metric='RMSE',
    od_type='Iter',
    allow_writing_files=False)
final_model.fit(train_x, train_y, silent=True)


daytime = time.localtime()

submission = pd.read_csv(
    "/Users/taehoon/Documents/dacon_movies/data/data/submission.csv", index_col=0)
submission["box_off_num"] = final_model.predict(test)
submission.to_csv("submission__"+"_".join(list(map(str, daytime)))+"__.csv")
