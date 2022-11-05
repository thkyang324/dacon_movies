import pandas as pd
from collections import defaultdict
path = "/Users/taehoon/Documents/dacon_movies/data/data/"


def load_data():
    test = pd.read_csv(path+"movies_test.csv")
    train = pd.read_csv(path+"movies_train.csv")
    return {"test": test, "train": train}


def process_time(data):
    data['release_year'] = data['release_time'].apply(
        lambda x: x.split('-')[0])
    data['release_month'] = data['release_time'].apply(
        lambda x: x.split('-')[1])
    data['release_day'] = data['release_time'].apply(
        lambda x: x.split('-')[2])
    data = data.drop("release_time", axis=1)
    return data


def get_director2value(data):
    temp = data.copy()
    temp['dir_prev_bfnum'] = temp["dir_prev_bfnum"].fillna(-1)
    temp['dir_prev_bfnum'] = temp["dir_prev_bfnum"].apply(str)
    groupby_director = temp.groupby('director')['dir_prev_bfnum'].apply(
        lambda x: '|'.join(x.replace("-1.0", ""))).reset_index()
    groupby_director["bfnum_sum"] = groupby_director["dir_prev_bfnum"].apply(
        lambda x: sum(list(map(float, list(filter(lambda t: t, ("0|"+x).split("|")))))))
    groupby_director["bfnum_len"] = groupby_director["dir_prev_bfnum"].apply(
        lambda x: len(list(map(float, list(filter(lambda t: t, ("0|"+x).split("|")))))))

    groupby_director["bfnum_avg"] = groupby_director["bfnum_sum"] / \
        groupby_director["bfnum_len"]
    director2value = defaultdict(int)
    for i, j in zip(groupby_director["director"], groupby_director["bfnum_avg"]):
        director2value[i] = j
    return director2value


def fill_bfnum_na(train, test):
    director2value = get_director2value(train)
    bfnum_nan_index = train["dir_prev_bfnum"].isna()
    train.loc[bfnum_nan_index, "dir_prev_bfnum"] = train.loc[bfnum_nan_index,
                                                             "director"].map(director2value)
    test_nan_index = test["dir_prev_bfnum"].isna()
    test.loc[test_nan_index, "dir_prev_bfnum"] = test.loc[test_nan_index,
                                                          "director"].map(director2value)
    return train, test


def load_preprocessed_data():
    train, test = load_data().values()
    train = process_time(train)
    test = process_time(test)
    train, test = fill_bfnum_na(train, test)
    return {"test": test, "train": train}
