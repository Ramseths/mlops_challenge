import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from mlflow import log_metric


def cross_validation(model_pipeline, X_train, y_train, n_splits, shuffle, random_state):
    kf = KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv = kf)
    return np.mean(cv_scores)

def get_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared = False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Log Metrics
    log_metric("RMSE", rmse)
    log_metric("MAE", mae)
    log_metric("R2", r2)
    return rmse, mae, r2