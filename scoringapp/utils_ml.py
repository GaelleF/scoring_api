import joblib
import os
import pickle
import lightgbm
import pandas as pd
from flask import jsonify
from lime import lime_tabular
import dill

my_dir = os.path.dirname("light_gbm.pkl")
#from models_pkl import light_gbm
#import models_pkl
# "SK_ID_CURR":100001.0,
data = [{"NAME_CONTRACT_TYPE":0.0,"CODE_GENDER":0.0,"FLAG_OWN_CAR":0.0,
"FLAG_OWN_REALTY":0.0,"CNT_CHILDREN":0.0,"AMT_INCOME_TOTAL":135000.0,"AMT_CREDIT":568800.0,
"REGION_RATING_CLIENT_W_CITY":2.0,"NEW_INCOME_CREDIT_RATIO":0.2373417722,
"NAME_INCOME_TYPE_Commercial associate":0.0,"NAME_INCOME_TYPE_Pensioner":0.0,
"NAME_INCOME_TYPE_State servant":0.0,"NAME_INCOME_TYPE_Working":1.0,
"NAME_EDUCATION_TYPE_Agriculture":0.0,"NAME_EDUCATION_TYPE_Business_Entity":0.0,
"NAME_EDUCATION_TYPE_Construction":0.0,"NAME_EDUCATION_TYPE_Education":1.0,
"NAME_EDUCATION_TYPE_Finance":0.0,"NAME_EDUCATION_TYPE_Government":0.0,"NAME_EDUCATION_TYPE_Industry":0.0,
"NAME_EDUCATION_TYPE_Official":0.0,"NAME_EDUCATION_TYPE_Other":0.0,"NAME_EDUCATION_TYPE_Realty":0.0,
"NAME_EDUCATION_TYPE_Secondary_secondary_special":0.0,"NAME_EDUCATION_TYPE_Security":0.0,
"NAME_EDUCATION_TYPE_Self-employed":0.0,"NAME_EDUCATION_TYPE_TourismFoodSector":0.0,
"NAME_EDUCATION_TYPE_Trade":0.0,"NAME_EDUCATION_TYPE_Transport":0.0,"NAME_EDUCATION_TYPE_XNA":0.0,
"NAME_FAMILY_STATUS_Civil marriage":0.0,"NAME_FAMILY_STATUS_Married":1.0,"NAME_FAMILY_STATUS_Separated":0.0,
"NAME_FAMILY_STATUS_Single_not_married":0.0,"NAME_FAMILY_STATUS_Widow":0.0,
"NAME_HOUSING_TYPE_Co-op apartment":0.0,"NAME_HOUSING_TYPE_House_apartment":1.0,
"NAME_HOUSING_TYPE_Municipal apartment":0.0,"NAME_HOUSING_TYPE_Office apartment":0.0,
"NAME_HOUSING_TYPE_Rented apartment":0.0,"NAME_HOUSING_TYPE_With parents":0.0,
"OCCUPATION_TYPE_Accountants":0.0,"OCCUPATION_TYPE_Core staff":0.0,"OCCUPATION_TYPE_Drivers":0.0,
"OCCUPATION_TYPE_High_skill_staff":0.0,"OCCUPATION_TYPE_Laborers":0.0,"OCCUPATION_TYPE_Low_skill_staff":0.0,
"OCCUPATION_TYPE_Managers":0.0,"OCCUPATION_TYPE_Medicine staff":0.0,"OCCUPATION_TYPE_Others":0.0,
"OCCUPATION_TYPE_Sales staff":0.0,"ORGANIZATION_TYPE_Agriculture":0.0,"ORGANIZATION_TYPE_Business_Entity":0.0,
"ORGANIZATION_TYPE_Construction":0.0,"ORGANIZATION_TYPE_Education":1.0,"ORGANIZATION_TYPE_Finance":0.0,
"ORGANIZATION_TYPE_Government":0.0,"ORGANIZATION_TYPE_Industry":0.0,"ORGANIZATION_TYPE_Official":0.0,
"ORGANIZATION_TYPE_Other":0.0,"ORGANIZATION_TYPE_Realty":0.0,"ORGANIZATION_TYPE_Security":0.0,
"ORGANIZATION_TYPE_Self-employed":0.0,"ORGANIZATION_TYPE_TourismFoodSector":0.0,
"ORGANIZATION_TYPE_Trade":0.0,"ORGANIZATION_TYPE_Transport":0.0,"ORGANIZATION_TYPE_XNA":0.0,
"NEW_SEGMENT_AGE_Middle_Age":1.0,"NEW_SEGMENT_AGE_Old":0.0,"NEW_SEGMENT_AGE_Young":0.0,
"NEW_SEGMENT_INCOME_High_Income":0.0,"NEW_SEGMENT_INCOME_Low_Income":0.0,
"NEW_SEGMENT_INCOME_Middle_Income":1.0}]


def model_predict():
  # return target + lime or shap

  pickle_path = os.getcwd()+"/scoringapp/models_pkl/"
  #model2 =  pickle.load(open(pickle_path), 'r')
  model = joblib.load(pickle_path + "light_gbm.pkl",'r')
  #explainer_light = dill.load(pickle_path + "explainer_light.pkl",'rb')
  #pickle_file_path = os.path.join(my_dir, "light_gbm.pkl")
  with open(pickle_path +"explainer_light.pkl", "rb") as f:
    explainer_light = dill.load(f)
    #model = joblib.load( f,'rb')
  # , lime:{....}
  response = model.predict(pd.DataFrame(data))
  print('RESSSSSSSSSS', response)
  exp_light = explainer_light.explain_instance(data[
    0], model.predict_proba,
  labels=response[0], num_features=5)

  #
  exp_light.as_pyplot_figure()
  return {'target' : response.tolist(), 'lime': 'test'}