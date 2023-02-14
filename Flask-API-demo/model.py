# -*- coding: UTF-8 -*-
import pickle
import gzip
import xgboost as xgb

# 載入Model
xgboostModel = xgb.XGBClassifier()
xgboostModel.load_model("./model/model_sklearn.json")


def predict(input):
    pred = xgboostModel.predict(input)[0]
    print(pred)
    return pred
