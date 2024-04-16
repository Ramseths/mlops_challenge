from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import  ElasticNet

def model_config(alpha, l1_ratio, max_iter, random_state):
    # ElasticNet Model
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