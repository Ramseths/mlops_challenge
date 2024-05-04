from preprocessing import load_data, preprocessing_data
import yaml
import pytest
import pandas as pd

with open("../config/params.yaml", "r") as f:
        params = yaml.safe_load(f)

@pytest.fixture
def data_params():
    return ("../data/Residential-Building-Data-Set.xlsx", "Data")      

def test_load_data(data_params):
    path, sheet_name = data_params
    df = load_data(path=path, sheet_name=sheet_name)
    assert isinstance(df, pd.DataFrame), "DataFrame"


    X_train, X_test, y_train, y_test = preprocessing_data(df, test_size=0.2, random_state=0)
    assert X_train.shape[1] > 0, "Features OK"


