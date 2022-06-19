import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


def prepare_data():
    data_set = pd.read_csv("data/insurance.csv", index_col=False)
    encoder = LabelEncoder()
    data_set.loc[:, "sex"] = encoder.fit_transform(data_set.loc[:, "sex"])
    # male = 1, female = 0
    data_set.loc[:, "smoker"] = encoder.fit_transform(data_set.loc[:, "smoker"])
    # yes = 1, no = 0
    data_set.loc[:, "region"] = encoder.fit_transform(data_set.loc[:, "region"])
    # NE = 0, NW =1, SE = 2, SW = 3
    data_set.to_csv('data/data_all.csv', index=False)
    df = pd.DataFrame(data_set)
    for s in range(0, len(df), 50):
        df.iloc[s:s + 50].to_csv(f"data/batch/batch{s // 50}.csv", index=False)


def train():
    data_set = pd.read_csv("data/data_all.csv", index_col=False)
    train_set, test_set = train_test_split(data_set, test_size=0.2)
    characteristics = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    X_train = train_set.loc[:, characteristics]
    y_train = train_set['charges']
    X_test = train_set.loc[:, characteristics]
    y_test = train_set['charges']
    reg = GradientBoostingRegressor(random_state=0)
    model = reg.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse_2 = np.sqrt(mean_squared_error(y_test, predictions))
    print('RMSE_2 = ', rmse_2)
    pickle.dump(model, open('model/model.pkl', 'wb'))


def retrain(alpha, learning_rate):
    data_set = pd.read_csv("data/data_all.csv", index_col=False)
    train_set, test_set = train_test_split(data_set, test_size=0.2)
    characteristics = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    X_train = train_set.loc[:, characteristics]
    y_train = train_set['charges']
    X_test = train_set.loc[:, characteristics]
    y_test = train_set['charges']
    reg = GradientBoostingRegressor(random_state=0, learning_rate=learning_rate, alpha=alpha)
    model = reg.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse_2 = np.sqrt(mean_squared_error(y_test, predictions))
    print('RMSE_2 = ', rmse_2)
    pickle.dump(model, open('model/model.pkl', 'wb'))
