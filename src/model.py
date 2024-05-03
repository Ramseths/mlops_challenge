from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import  ElasticNet
from joblib import dump
from mlflow import log_param, sklearn

def model_config(alpha, l1_ratio, max_iter, random_state):
    # ElasticNet Model
    log_param("alpha", alpha)
    log_param("l1_ratio", l1_ratio)
    log_param("max_iter", max_iter)
    log_param("random_state", random_state)
    model = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = max_iter, random_state = random_state)
    return model


def train_pipeline(X_train, y_train, scaler, features, model):
    # Transformer
    transformer = PowerTransformer(method='yeo-johnson')

    # Pipeline
    feature_pipeline = Pipeline([
        ('yeo_johnson', transformer),
        ('scaler', scaler)
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('features', feature_pipeline, features)
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('model', model)
    ])

    # Train model
    model_pipeline.fit(X_train, y_train)

    return model_pipeline


def model_save(model, path, version):
    """
    This function helps to save model.

    Parameters
    ----------
    model: Scikitlearn model or pipeline.
    path: Path to save model.
    version: Model version.
    """

    try:
        # Save local model with dump
        dump(model, path + f'trained_model-{version}.joblib')
        # Save artifact
        sklearn.log_model(model, "trained_model", registered_model_name = "trained_model" + version)
        print("Successfully Model Saved")
    except Exception as e:
        print(f"Error Model Save: {e}")
