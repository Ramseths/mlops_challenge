import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path, sheet_name):
    """
    This function helps to load data from DVC.

    Parameters
    ----------
    path: str
    sheet_name: str

    Returns
    -------
    df: DataFrame
    """
    try:
        df = pd.read_excel(path, sheet_name = sheet_name, skiprows = 1)
        return df
    except Exception as e:
        print(f'Error load_data {e}')

def preprocessing_data(data, test_size, random_state):
    """
    This function helps to prepare data to train model.

    Parameters
    ----------
    data: DataFrame
    test_size: float
    random_state: int

    Returns
    -------

    """
    # Without lags
    cols = ['V-2', 'V-20', 'V-25', 'V-9']
    data =  data[cols].copy()

    # Rename cols
    name_cols = ['Floor Area', 'Interest', 'CPI', 'Price']
    data.columns = name_cols

    X = data.drop('Price', axis = 1) 
    y = data['Price'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    print(f'Train Data (Shape): X - {X_train.shape}, y - {y_train.shape}')
    print(f'Test Data (Shape): X {X_test.shape}, y - {y_test.shape}')

    return X_train, X_test, y_train, y_test