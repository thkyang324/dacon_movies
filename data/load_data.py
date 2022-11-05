import pandas as pd


def month2season(val):
    val = int(val)
    if val in (1, 7, 8, 9, 12):
        return 1
    return 0


def load_data(data):
    data['release_year'] = data['release_time'].apply(
        lambda x: x.split('-')[0])
    data['release_month'] = data['release_time'].apply(
        lambda x: x.split('-')[1])
    data['release_day'] = data['release_time'].apply(lambda x: x.split('-')[2])
    data['season'] = data['release_month'].apply(month2season)
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
    director2bfnum_avg = {i: j for i, j in zip(
        groupby_director["director"], groupby_director["bfnum_avg"])}
    bfnum_nan_index = data["dir_prev_bfnum"].isna()
    data.loc[bfnum_nan_index, "dir_prev_bfnum"] = data.loc[bfnum_nan_index,
                                                           "director"].map(director2bfnum_avg)
    data = data.drop("release_time", axis=1)
    return data


def load_preprocessed_data(path=""):
    train = load_data(pd.read_csv(path+"movies_test.csv"))
    test = load_data(pd.read_csv(path+"movies_train.csv"))
    return train, test


if __name__ == "__main__":
    print(load_preprocessed_data())
