from src.data.preprocessing import load_preprocessed_data
import pandas as pd
import sklearn
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data = load_preprocessed_data()

train = data["train"]
test = data["test"]


def label_encoding(data):
    target = data["train"][['box_off_num']]
    train = data["train"].drop('box_off_num', axis=1)
    encoder = ce.cat_boost.CatBoostEncoder()
    encoder.fit(train, target)
    train_cbe = encoder.transform(train)
    test_cbe = encoder.transform(test)
    return {"train": train_cbe, "test": test_cbe}


label_encoding(data)
