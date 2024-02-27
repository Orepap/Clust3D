import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


def impute(df, imputation):

    if imputation == "none":
        df = df

    elif imputation == "zeros":
        df.fillna(0, inplace=True)

    elif imputation == "median":
        imputer = SimpleImputer(strategy='median')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    elif imputation == "knn":
        imputer = KNNImputer(n_neighbors=10)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    else:
        print("ERROR: Enter 'none', 'zeros', 'median' or 'knn' for the 'imputation' parameter value")
        exit()

    return df
