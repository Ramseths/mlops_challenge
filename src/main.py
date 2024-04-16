import yaml
from preprocessing import load_data, preprocessing_data
from model import model_config, train_pipeline
from evaluation import cross_validation, get_metrics
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    
    # Load config
    with open("../config/params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data = load_data(path = params['data']['path'], sheet_name = params['data']['sheet_name'])
    X_train, X_test, y_train, y_test = preprocessing_data(data=data, test_size=params["data"]["test_size"], 
                                                        random_state=params["data"]["random_state"])
    
    model = model_config(
        alpha=params['model']['alpha'],
        l1_ratio=params['model']['l1_ratio'],
        max_iter=params['model']['max_iter'],
        random_state=params["data"]["random_state"]
    )

    model_pipeline = train_pipeline(
        X_train=X_train,
        y_train=y_train,
        scaler=StandardScaler(),
        features=X_train.columns.to_list(),
        model=model
    )

    y_pred = model_pipeline.predict(X_test)
    mean_cv_scores = cross_validation(
        model_pipeline=model_pipeline,
        X_train=X_train,
        y_train=y_train, 
        n_splits=params["evaluation"]["n_splits"],
        shuffle=params["evaluation"]["shuffle"],
        random_state=params["data"]["random_state"]
    )

    # Final metrics
    rmse, mae, r2 = get_metrics(y_test, y_pred)

    print(f'Mean Cross Validation Score: {mean_cv_scores}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R2: {r2}')

