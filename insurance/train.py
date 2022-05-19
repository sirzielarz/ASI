import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        data = pd.read_csv("insurance.csv", sep=",")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    print(data)
    # data.drop('region', inplace=True, axis=1)
    encoder = LabelEncoder()
    data.loc[:, "sex"] = encoder.fit_transform(data.loc[:, "sex"])
    # male = 1, female = 0
    data.loc[:, "smoker"] = encoder.fit_transform(data.loc[:, "smoker"])
    # yes = 1, no = 0
    data.loc[:, "region"] = encoder.fit_transform(data.loc[:, "region"])
    # NE = 0, NW =1, SE = 2, SW = 3

    data.to_csv('data/data_all.csv',index=False)


    print(data)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, rest = train_test_split(data,train_size=0.45, test_size=0.55)
    test, rest2 = train_test_split(rest,train_size=0.65, test_size=0.35)
    validation, production = train_test_split(rest2,train_size=0.55, test_size=0.45)
    train.to_csv('data/data_train.csv', index=False)
    test.to_csv('data/data_test.csv', index=False)
    validation.to_csv('data/data_validation.csv', index=False)
    production.to_csv('data/data_production.csv', index=False)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["charges"], axis=1)
    test_x = test.drop(["charges"], axis=1)
    train_y = train[["charges"]]
    test_y = test[["charges"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
