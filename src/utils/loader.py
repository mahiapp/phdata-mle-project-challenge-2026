import json
import pickle

import pandas as pd

from utils.imputation import get_imputer

model = None
model_features = None
demographics = None


def load_model(model_path: str):
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def load_features(features_path: str):
    with open(features_path, "r") as features_file:
        return json.load(features_file)


def load_resources():
    global model, model_features, demographics
    model = load_model("model/model.pkl")
    model_features = load_features("model/model_features.json")
    demographics = pd.read_csv("data/zipcode_demographics.csv", dtype={"zipcode": str})
    get_imputer()


load_resources()
