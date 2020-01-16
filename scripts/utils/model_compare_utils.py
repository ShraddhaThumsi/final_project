import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from math import sqrt
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.metrics as metrics
import utils.dataframe_utils as df_utils
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import time
import json




#Thumsi will handle Decision Tree classifier, Gaussian Process Classifier and Adaboost.
#models that Salas will work on are: XGBoost, LightGBM, CatBoost

def get_classifier_by_name(name_of_classifier, num_of_features=None):
    '''
    This function brings the name of the classifier from the list of classifiers implemented for this project.
    :param name_of_classifier:
    :param num_of_features:
    :return:
    '''
    if "xg_boost" in name_of_classifier:
        # this classifier is available in Scala also
        return get_xgb_classifier()


    if "light_gbm" in name_of_classifier:
        return get_lightgbm_classifier()

    if "cat_boost" in name_of_classifier:
        return get_catboost_classifier()

    if "decision_tree" in name_of_classifier:
        return get_decision_tree_classifier(num_of_features)

    if "gaussian_process" in name_of_classifier:
        return get_gaussian_classifier()

    if "adaboost" in name_of_classifier:
        return get_adaboost_classifier()

def get_folds(dataframe):
    '''
    Takes data frame as input and returns lists ot tuples as output.
    This reads the paper IDs in the data frame and create false cross validation
    We loop through the list of paper IDs in this data frame and create a tuple where the current paper ID is the test
    case and all the other paper ID are training case
    :param dataframe:
    :return:
    '''
    folds = []
    papers_in_frame = set(list(dataframe["PMCID"]))
    for p in papers_in_frame:
        test_case = dataframe[dataframe["PMCID"] == p]
        train_case = dataframe[dataframe["PMCID"]!=p]
        folds.append((test_case,train_case))
    return folds

def get_true_pred_arrays_from_cv(classifier,dataframe,is_classifier_catboost=False):
    '''
    This function takes the classifier and data frame as parameters.
    It trains the classifier on the training set and tests it on the testing set in a cross validation setting. We then
    return a tuple of true labels and predicted labels
    :param classifier:
    :param dataframe:
    :param is_classifier_catboost:
    :return:
    '''
    fold_dfs = get_folds(dataframe)
    per_classifier_true_array = []
    per_classifier_predicted_array = []
    for test_df,train_df in fold_dfs:
        balanced_train_dfs = df_utils.get_balanced_dataframe(train_df,1) # Getting same number of false and true labels
        data_features_only = get_data_features_only(balanced_train_dfs)
        if is_classifier_catboost is True:
            # We do this because CatBoost wants to take the label array as a list of numbers rather than list of booleans.
            # To fix this, we convert the boolean label array to an int array which will be taken care of by the model automatically.
            # Everything else remains the same.
            datavalues_from_train_set, _,labels_from_train_set = extract_feature_values_from_df(balanced_train_dfs,data_features_only)
            datavalues_from_test_set, _, labels_from_test_set = extract_feature_values_from_df(balanced_train_dfs,
                                                                                                 data_features_only)

        else:
            datavalues_from_train_set,labels_from_train_set,_ = extract_feature_values_from_df(balanced_train_dfs,data_features_only)
            datavalues_from_test_set,labels_from_test_set,_ = extract_feature_values_from_df(test_df,data_features_only)
        classifier.fit(datavalues_from_train_set,labels_from_train_set)
        # here we convert the int label array from catBoost back to a Boolean array to keep parity with the other algorithms.
        predicted_test_array = [bool(i) for i in classifier.predict(datavalues_from_test_set)]
        true_test_array = [bool(i) for i in labels_from_test_set]
        per_classifier_true_array.extend(true_test_array)
        per_classifier_predicted_array.extend(predicted_test_array)
    return per_classifier_true_array,per_classifier_predicted_array


def get_true_preds_arrays_per_classifier(names_of_classifiers, dataframe):
    '''
    This function takes the list of classifiers and the data frame as parameters.
    For each of the classifiers it trains and predicts on the data frame in a cross validation setting. Finally, we
    return a dictionary where the name of the classifier is the key and a tuple of true predicted labels is the value.
    :param names_of_classifiers:
    :param dataframe:
    :return:
    '''
    per_classifier_truepred_arrays = dict()
    per_classifier_time_consumption_minutes = dict()
    for c in names_of_classifiers:
        features_in_dataframe = list(dataframe.columns.values)
        current_classifier_to_use = get_classifier_by_name(c,len(features_in_dataframe))
        start_time = time.time()

        if "cat" in c:
            (true_array, predicted_array) = get_true_pred_arrays_from_cv(current_classifier_to_use, dataframe,is_classifier_catboost=True)
        else:
            (true_array,predicted_array) = get_true_pred_arrays_from_cv(current_classifier_to_use, dataframe)
        print(f"finished {c}")
        per_classifier_truepred_arrays[c] = (true_array,predicted_array)
        per_classifier_time_consumption_minutes[c] = format_num((time.time() - start_time)/60)

    return per_classifier_truepred_arrays, per_classifier_time_consumption_minutes

def format_num(num):
    return '{0:.3g}'.format(num)

def get_scores_per_classifier(per_class_pred_list_dict):
    '''
    In this function we take the dictionary of classifier name -> (true array, predicted array). We calculate the f1,
    precision, recall and accuracy scores using the scikitlearn.metrics package
    :param per_class_pred_list_dict:
    :return:
    '''
    per_class_score_dict = dict()
    for classifier_name in per_class_pred_list_dict.keys():
        true_array,pred_array = per_class_pred_list_dict[classifier_name]
        f1 = metrics.f1_score(true_array,pred_array)
        accuracy = metrics.accuracy_score(true_array,pred_array)
        precision = metrics.precision_score(true_array,pred_array)
        recall = metrics.recall_score(true_array,pred_array)
        score_dict = {"f1":format_num(f1), "precision":format_num(precision),"recall":format_num(recall),"accuracy":format_num(accuracy)}
        per_class_score_dict[classifier_name] = score_dict

    return per_class_score_dict



def write_classifier_scores_to_file(per_classifier_score_dict, path_to_json_file):
    with open(path_to_json_file,"w") as file:
        json.dump(per_classifier_score_dict,file)



def write_classifier_time_consumption_to_file(per_classifier_time_dict,path_to_json_file):
    with open(path_to_json_file,"w") as file:
        json.dump(per_classifier_time_dict,file)

def extract_feature_values_from_df(df, features):
    '''
    This function takes the data frame and list of features as parameters . It returns a tuple of data feature values
    labeled as boolean values and labeled as integer values.
    :param df:
    :param features:
    :return:
    '''
    X = df[features]
    X = X.values if len(features) > 1 else X.values.reshape((X.size, 1))

    y = df['label'].values.astype("bool")
    y_as_int = [int(i) for i in list(y)]

    return X, y, y_as_int

def get_data_features_only(df):
    '''
    This function takes the data frame as parameters and returns a list of features. These features were found to perform
    best in a ablation study conducted in 2018. Please contact Shraddha for further details.
    :param df:
    :return:
    '''
    meta_features = ["PMCID", "EvtID", "CtxID", "label", "Unnamed: 0"]
    data_values = list(set(df.columns.values) - set(meta_features))
    return data_values


# creating instances of the machine learning models.
def get_decision_tree_classifier(num_of_feats,depth=8):
    '''
    This function creates the instances for the decision tree classifier.
    :param num_of_feats:
    :param depth:
    :return:
    '''
    empirical_max = int(sqrt(num_of_feats))
    return DecisionTreeClassifier(max_depth=depth,
                                  max_features=empirical_max)

def get_adaboost_classifier():
    '''
    This function creates the instances for the adaboost classifier
    :return:
    '''
    est = 32
    return AdaBoostClassifier(n_estimators=est)


def get_gaussian_classifier():
    '''
    This function creates the instances for the gaussian classifier
    :return:
    '''
    return GaussianProcessClassifier()

def get_xgb_classifier():
    '''
    This function creates the instances for the xgb classifier
    :return:
    '''
    return xgb.XGBClassifier()

def get_catboost_classifier():
    ''''
    This function creates the instances for the catboat classifier
    '''
    return CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1, loss_function='Logloss')

def get_lightgbm_classifier():
    '''
    This function creates the instances for the light GBM classifier
    :return:
    '''
    return LGBMClassifier(objective='binary', random_state=5)

