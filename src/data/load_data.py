import pandas as pd
import os


def month2season(val):
    val = int(val)
    if val in (1, 7, 8, 9, 12):
        return 1
    return 0


def getgenre2idx():
    df = pd.read_csv(
        "/Users/taehoon/Documents/dacon_movies/data/data/movies_train.csv")
    genre_encoding = df[['genre', 'box_off_num']].groupby(
        'genre').mean().sort_values('box_off_num').reset_index()
    genre2idx = {i: j for i, j in zip(
        genre_encoding["genre"], genre_encoding["box_off_num"])}
    return genre2idx


def getscreening_rat2idx():
    df = pd.read_csv(
        "/Users/taehoon/Documents/dacon_movies/data/data/movies_train.csv")
    screening_rat_encoding = df[['screening_rat', 'box_off_num']].groupby(
        'screening_rat').mean().sort_values('box_off_num').reset_index()
    screening_rat2idx = {i: j for i, j in zip(
        screening_rat_encoding["screening_rat"], screening_rat_encoding["box_off_num"])}
    return screening_rat2idx


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
    director2bfnum_len = {i: j for i, j in zip(
        groupby_director["director"], groupby_director["bfnum_len"])}
    director2bfnum_sum = {i: j for i, j in zip(
        groupby_director["director"], groupby_director["bfnum_sum"])}

    bfnum_nan_index = data["dir_prev_bfnum"].isna()
    data.loc[bfnum_nan_index, "dir_prev_bfnum"] = data.loc[bfnum_nan_index,
                                                           "director"].map(director2bfnum_avg)
    data["dir_prev_bfnum_len"] = data["director"].map(director2bfnum_len)
    data["dir_prev_bfnum_sum"] = data["director"].map(director2bfnum_sum)

    genre2idx = getgenre2idx()
    screening_rat2idx = getscreening_rat2idx()
    data["new_genre"] = data["genre"].map(genre2idx)
    data["new_screening_rat"] = data["screening_rat"].map(screening_rat2idx)

    data = data.drop("release_time", axis=1)
    return data


def load_preprocessed_data(path="/Users/taehoon/Documents/dacon_movies/data/data"):
    train = load_data(pd.read_csv(path+"/movies_train.csv"))
    test = load_data(pd.read_csv(path+"/movies_test.csv"))
    temp = train["box_off_num"]
    train = train.drop("box_off_num", axis=1)
    train["box_off_num"] = temp
    return train, test


if __name__ == "__main__":
    savepath = "/Users/taehoon/Documents/dacon_movies/data/data"
    folder_name = "preprocessed_data"
    train, test = load_preprocessed_data()
    if folder_name not in os.listdir(savepath):
        os.mkdir(savepath+"/"+folder_name)
        print("*"*20 + "    make directory preprocessed_data   " + "*"*20)
    train.to_csv(savepath+"/"+folder_name+"/preprocessed_train.csv")
    test.to_csv(savepath+"/"+folder_name+"/preprocessed_test.csv")
