import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def run(model, batch):
    model_name = model
    model = pickle.load(open(model_name, 'rb'))
    batch_no = batch
    test_data = pd.read_csv("data/batch/batch" + str(batch_no) + ".csv")
    characteristics = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    X = test_data.loc[:, characteristics]
    y = test_data['charges']
    predictions = model.predict(X)
    RMSE = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    eval_df = pd.DataFrame()
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    eval_df = eval_df.append({'time_stamp': now, 'version': '1.0', 'batch': batch_no, 'metric': 'RMSE', 'score': RMSE},
                             ignore_index=True)
    eval_df = eval_df.append({'time_stamp': now, 'version': '1.0', 'batch': batch_no, 'metric': 'r2', 'score': r2},
                             ignore_index=True)

    evaluation_file_name = 'evaluation/model_eval.csv'

    if os.path.isfile(evaluation_file_name):
        eval_df.to_csv('evaluation/model_eval.csv', mode='a', index=False, header=False)
    else:
        eval_df.to_csv('evaluation/model_eval.csv', index=False)
