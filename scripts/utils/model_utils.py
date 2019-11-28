import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#Thumsi will handle Random forest and LSTM.
#models I will compare here are:
# XGBoost, XGLite, XGBM

def get_classifier_by_name(name_of_classifier):
    if "xg_boost" in name_of_classifier:
        return
    if "random_forest" in name_of_classifier:
        return

def get_folds(dataframe):
    folds = []
    papers_in_frame = set(list(dataframe["PMCID"]))
    for p in papers_in_frame:
        test_case = dataframe[dataframe["PMCID"] == p]
        train_case = dataframe[dataframe["PMCID"]!=p]
        folds.append((test_case,train_case))
    return folds


