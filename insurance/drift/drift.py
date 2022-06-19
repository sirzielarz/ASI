import random
import numpy as np
import pandas as pd
from datetime import datetime
import os.path
from insurance.model import read_and_train


def detect():
    eval_results = pd.read_csv('evaluation/model_eval.csv', parse_dates=['time_stamp'], dayfirst=True)

    last_run = eval_results['time_stamp'].max()
    model_version = eval_results[eval_results['time_stamp'] == last_run]['version'].values[0]

    # Prepare data for tests
    RMSE_logs = eval_results[eval_results['metric'] == 'RMSE']
    r2_logs = eval_results[eval_results['metric'] == 'r2']

    last_RMSE = RMSE_logs[RMSE_logs['time_stamp'] == last_run]['score'].values[0]
    all_other_RMSE = RMSE_logs[RMSE_logs['time_stamp'] != last_run]['score'].values

    last_r2 = r2_logs[r2_logs['time_stamp'] == last_run]['score'].values[0]
    all_other_r2 = r2_logs[r2_logs['time_stamp'] != last_run]['score'].values

    ### Hard test ###
    hard_test_RMSE = last_RMSE > np.mean(all_other_RMSE)
    hard_test_r2 = last_r2 < np.mean(all_other_r2)
    print('\nLegend: \nTRUE means the model has drifted. FALSE means the model has not.')
    print('\n.. Hard test ..')
    print('RMSE: ', hard_test_RMSE, '  R2: ', hard_test_r2)
    param_test_RMSE = last_RMSE > np.mean(all_other_RMSE) + 2 * np.std(all_other_RMSE)
    param_test_r2 = last_r2 < np.mean(all_other_r2) - 2 * np.std(all_other_r2)
    print('\n.. Parametric test ..')
    print('RMSE: ', param_test_RMSE, '  R2: ', param_test_r2)
    iqr_RMSE = np.quantile(all_other_RMSE, 0.75) - np.quantile(all_other_RMSE, 0.25)
    iqr_test_RMSE = last_RMSE > np.quantile(all_other_RMSE, 0.75) + iqr_RMSE * 1.5
    iqr_r2 = np.quantile(all_other_r2, 0.75) - np.quantile(all_other_r2, 0.25)
    iqr_test_r2 = last_r2 < np.quantile(all_other_r2, 0.25) - iqr_r2 * 1.5
    print('\n.. IQR test ..')
    print('RMSE: ', iqr_test_RMSE, '  R2: ', iqr_test_r2)
    drift_df = pd.DataFrame()
    drift_signal_file = 'evaluation/model_drift.csv'
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    print('\n  --- DRIFT DETECTION ---')
    actual_tests = {
        'hard_test_RMSE': hard_test_RMSE,
        'hard_test_r2': hard_test_r2,
        'param_test_RMSE': param_test_RMSE,
        'param_test_r2': param_test_r2,
        'iqr_test_RMSE': iqr_test_RMSE,
        'iqr_test_r2': iqr_test_r2
    }
    a_set = set(actual_tests.values())
    drift_detected = False
    if True in set(actual_tests.values()):
        drift_detected = True
    if drift_detected:
        print('There is a DRIFT detected in...')
        for a in actual_tests:
            if actual_tests[a]:
                print(a)
        drift_df = drift_df.append({'time_stamp': now, 'model_name': model_version,
                                    'hard_test_RMSE': str(hard_test_RMSE),
                                    'hard_test_r2': str(hard_test_r2),
                                    'param_test_RMSE': str(param_test_RMSE),
                                    'param_test_r2': str(param_test_r2),
                                    'iqr_test_RMSE': str(iqr_test_RMSE),
                                    'iqr_test_r2': str(iqr_test_r2)
                                    }, ignore_index=True)
        if os.path.isfile(drift_signal_file):
            drift_df.to_csv(drift_signal_file, mode='a', header=False, index=False)
        else:
            drift_df.to_csv(drift_signal_file, index=False)
    else:
        print('There is NO DRIFT detected.')
    if drift_detected:
        print('\n  --- RE-TRAINING ---\n')
        read_and_train.retrain(random.uniform(0.1, 0.9), random.uniform(0.1, 0.5))
