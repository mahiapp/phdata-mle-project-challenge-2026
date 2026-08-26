import pandas as pd
from sklearn.impute import KNNImputer

HOME_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
]

_imputer = None


def get_imputer():
    global _imputer
    if _imputer is None:
        data = pd.read_csv("data/kc_house_data.csv", usecols=HOME_FEATURES)
        _imputer = KNNImputer(n_neighbors=5, weights="distance")
        _imputer.fit(data)
    return _imputer


def impute_home_features(payload: dict) -> pd.DataFrame:
    row = {col: payload.get(col) for col in HOME_FEATURES}
    frame = pd.DataFrame([row], columns=HOME_FEATURES)
    values = get_imputer().transform(frame)
    return pd.DataFrame(values, columns=HOME_FEATURES)
