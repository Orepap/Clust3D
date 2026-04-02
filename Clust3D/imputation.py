import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


def impute(df, imputation):
    """
    Apply imputation method to a DataFrame with missing values.

    Parameters:
        df (pd.DataFrame): Input data.
        imputation (str): Method to use: 'none', 'zeros', 'median', or 'knn'.

    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    if imputation == "none":
        return df

    elif imputation == "zeros":
        df.fillna(0, inplace=True)
        return df

    elif imputation == "median":
        imputer = SimpleImputer(strategy='median')
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    elif imputation == "knn":
        imputer = KNNImputer(n_neighbors=10)
        return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    else:
        print("ERROR: Enter 'none', 'zeros', 'median' or 'knn' for the 'imputation' parameter value")
        exit()
