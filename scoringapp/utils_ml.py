import joblib
import os
import pickle
import lightgbm
import pandas as pd
import numpy as np
from flask import jsonify, json
from lime import lime_tabular
import dill
from sklearn.preprocessing import StandardScaler

my_dir = os.path.dirname("light_gbm.pkl")
# from models_pkl import light_gbm
# import models_pkl
# "SK_ID_CURR":100001.0,
data_test = [
    {
        "NAME_CONTRACT_TYPE": 0.0,
        "CODE_GENDER": 0.0,
        "FLAG_OWN_CAR": 0.0,
        "FLAG_OWN_REALTY": 0.0,
        "CNT_CHILDREN": 0.0,
        "AMT_INCOME_TOTAL": 135000.0,
        "AMT_CREDIT": 568800.0,
        "REGION_RATING_CLIENT_W_CITY": 2.0,
        "NEW_INCOME_CREDIT_RATIO": 0.2373417722,
        "NAME_INCOME_TYPE_Commercial associate": 0.0,
        "NAME_INCOME_TYPE_Pensioner": 0.0,
        "NAME_INCOME_TYPE_State servant": 0.0,
        "NAME_INCOME_TYPE_Working": 1.0,
        "NAME_EDUCATION_TYPE_Agriculture": 0.0,
        "NAME_EDUCATION_TYPE_Business_Entity": 0.0,
        "NAME_EDUCATION_TYPE_Construction": 0.0,
        "NAME_EDUCATION_TYPE_Education": 1.0,
        "NAME_EDUCATION_TYPE_Finance": 0.0,
        "NAME_EDUCATION_TYPE_Government": 0.0,
        "NAME_EDUCATION_TYPE_Industry": 0.0,
        "NAME_EDUCATION_TYPE_Official": 0.0,
        "NAME_EDUCATION_TYPE_Other": 0.0,
        "NAME_EDUCATION_TYPE_Realty": 0.0,
        "NAME_EDUCATION_TYPE_Secondary_secondary_special": 0.0,
        "NAME_EDUCATION_TYPE_Security": 0.0,
        "NAME_EDUCATION_TYPE_Self-employed": 0.0,
        "NAME_EDUCATION_TYPE_TourismFoodSector": 0.0,
        "NAME_EDUCATION_TYPE_Trade": 0.0,
        "NAME_EDUCATION_TYPE_Transport": 0.0,
        "NAME_EDUCATION_TYPE_XNA": 0.0,
        "NAME_FAMILY_STATUS_Civil marriage": 0.0,
        "NAME_FAMILY_STATUS_Married": 1.0,
        "NAME_FAMILY_STATUS_Separated": 0.0,
        "NAME_FAMILY_STATUS_Single_not_married": 0.0,
        "NAME_FAMILY_STATUS_Widow": 0.0,
        "NAME_HOUSING_TYPE_Co-op apartment": 0.0,
        "NAME_HOUSING_TYPE_House_apartment": 1.0,
        "NAME_HOUSING_TYPE_Municipal apartment": 0.0,
        "NAME_HOUSING_TYPE_Office apartment": 0.0,
        "NAME_HOUSING_TYPE_Rented apartment": 0.0,
        "NAME_HOUSING_TYPE_With parents": 0.0,
        "OCCUPATION_TYPE_Accountants": 0.0,
        "OCCUPATION_TYPE_Core staff": 0.0,
        "OCCUPATION_TYPE_Drivers": 0.0,
        "OCCUPATION_TYPE_High_skill_staff": 0.0,
        "OCCUPATION_TYPE_Laborers": 0.0,
        "OCCUPATION_TYPE_Low_skill_staff": 0.0,
        "OCCUPATION_TYPE_Managers": 0.0,
        "OCCUPATION_TYPE_Medicine staff": 0.0,
        "OCCUPATION_TYPE_Others": 0.0,
        "OCCUPATION_TYPE_Sales staff": 0.0,
        "ORGANIZATION_TYPE_Agriculture": 0.0,
        "ORGANIZATION_TYPE_Business_Entity": 0.0,
        "ORGANIZATION_TYPE_Construction": 0.0,
        "ORGANIZATION_TYPE_Education": 1.0,
        "ORGANIZATION_TYPE_Finance": 0.0,
        "ORGANIZATION_TYPE_Government": 0.0,
        "ORGANIZATION_TYPE_Industry": 0.0,
        "ORGANIZATION_TYPE_Official": 0.0,
        "ORGANIZATION_TYPE_Other": 0.0,
        "ORGANIZATION_TYPE_Realty": 0.0,
        "ORGANIZATION_TYPE_Security": 0.0,
        "ORGANIZATION_TYPE_Self-employed": 0.0,
        "ORGANIZATION_TYPE_TourismFoodSector": 0.0,
        "ORGANIZATION_TYPE_Trade": 0.0,
        "ORGANIZATION_TYPE_Transport": 0.0,
        "ORGANIZATION_TYPE_XNA": 0.0,
        "NEW_SEGMENT_AGE_Middle_Age": 1.0,
        "NEW_SEGMENT_AGE_Old": 0.0,
        "NEW_SEGMENT_AGE_Young": 0.0,
        "NEW_SEGMENT_INCOME_High_Income": 0.0,
        "NEW_SEGMENT_INCOME_Low_Income": 0.0,
        "NEW_SEGMENT_INCOME_Middle_Income": 1.0,
    }
]


class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(
            s
        )  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str) or isinstance(o, unicode):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o


def model_predict(payload):
    # return target + lime or shap

    data = {}
    for key in payload:
        if key in [
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "REGION_RATING_CLIENT_W_CITY",
            "NEW_INCOME_CREDIT_RATIO",
            "NAME_INCOME_TYPE_Commercial associate",
            "NAME_INCOME_TYPE_Pensioner",
            "NAME_INCOME_TYPE_State servant",
            "NAME_INCOME_TYPE_Working",
            "NAME_EDUCATION_TYPE_Higher education",
            "NAME_EDUCATION_TYPE_Incomplete higher",
            "NAME_EDUCATION_TYPE_Lower secondary",
            "NAME_EDUCATION_TYPE_Secondary_secondary_special",
            "NAME_FAMILY_STATUS_Civil marriage",
            "NAME_FAMILY_STATUS_Married",
            "NAME_FAMILY_STATUS_Separated",
            "NAME_FAMILY_STATUS_Single_not_married",
            "NAME_FAMILY_STATUS_Widow",
            "NAME_HOUSING_TYPE_Co-op apartment",
            "NAME_HOUSING_TYPE_House_apartment",
            "NAME_HOUSING_TYPE_Municipal apartment",
            "NAME_HOUSING_TYPE_Office apartment",
            "NAME_HOUSING_TYPE_Rented apartment",
            "NAME_HOUSING_TYPE_With parents",
            "OCCUPATION_TYPE_Accountants",
            "OCCUPATION_TYPE_Core staff",
            "OCCUPATION_TYPE_Drivers",
            "OCCUPATION_TYPE_High_skill_staff",
            "OCCUPATION_TYPE_Laborers",
            "OCCUPATION_TYPE_Low_skill_staff",
            "OCCUPATION_TYPE_Managers",
            "OCCUPATION_TYPE_Medicine staff",
            "OCCUPATION_TYPE_Others",
            "OCCUPATION_TYPE_Sales staff",
            "ORGANIZATION_TYPE_Agriculture",
            "ORGANIZATION_TYPE_Business_Entity",
            "ORGANIZATION_TYPE_Construction",
            "ORGANIZATION_TYPE_Education",
            "ORGANIZATION_TYPE_Finance",
            "ORGANIZATION_TYPE_Government",
            "ORGANIZATION_TYPE_Industry",
            "ORGANIZATION_TYPE_Official",
            "ORGANIZATION_TYPE_Other",
            "ORGANIZATION_TYPE_Realty",
            "ORGANIZATION_TYPE_Security",
            "ORGANIZATION_TYPE_Self-employed",
            "ORGANIZATION_TYPE_TourismFoodSector",
            "ORGANIZATION_TYPE_Trade",
            "ORGANIZATION_TYPE_Transport",
            "ORGANIZATION_TYPE_XNA",
            "NEW_SEGMENT_AGE_Middle_Age",
            "NEW_SEGMENT_AGE_Old",
            "NEW_SEGMENT_AGE_Young",
            "NEW_SEGMENT_INCOME_High_Income",
            "NEW_SEGMENT_INCOME_Low_Income",
            "NEW_SEGMENT_INCOME_Middle_Income",
        ]:
            data[key] = float(payload[key])
    pickle_path = os.getcwd() + "/scoringapp/models_pkl/"
    # data=[]
    model = joblib.load(pickle_path + "lightGBM.pkl", "r")
    standardisation = joblib.load(pickle_path + "std_scale.pkl", "r")

    with open(pickle_path + "lightGBM_lime.pkl", "rb") as f:
        explainer_light = dill.load(f)
    print(data)
    data_format = standardisation.transform(pd.DataFrame([data]))
    response = model.predict(data_format)

    exp_light = explainer_light.explain_instance(
        pd.Series(data), model.predict_proba, num_features=6
    )

    return {"target": response.tolist(), "lime": exp_light.as_list()}
