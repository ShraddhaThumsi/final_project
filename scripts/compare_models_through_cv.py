import utils.dataframe_utils as df_utils
import utils.model_compare_utils as model_utils
import paths_in_final_project as paths



PATH_T0_DATAFRAME = paths.PATH_TO_DATAFRAME
print(PATH_T0_DATAFRAME)
dataframe = df_utils.get_dataframe_from_csv(PATH_T0_DATAFRAME)

#list_of_models_to_run = ["xg_boost","light_gbm","cat_boost","decision_tree","gaussian_process","adaboost"]
#list_of_models_to_run = ["decision_tree","adaboost", "cat_boost","gaussian_process"]
list_of_models_to_run=["cat_boost"]
#list_of_models_to_run = ["xg_boost","light_gbm","cat_boost","decision_tree","gaussian_process","adaboost"]
(truepred_dict_per_classifier,time_in_minutes_per_classifier) = model_utils.get_true_preds_arrays_per_classifier(list_of_models_to_run, dataframe)
scores_per_classifier = model_utils.get_scores_per_classifier(truepred_dict_per_classifier)
for classifier in scores_per_classifier.keys():
    score_dict = scores_per_classifier[classifier]
    print(f"Current model name is: {classifier}")
    print(f"Scores for this model: ")
    print(score_dict)

PATH_TO_SCORES_JSON = paths.PATH_TO_SCORES_JSON
model_utils.write_classifier_scores_to_file(scores_per_classifier,PATH_TO_SCORES_JSON)
PATH_TO_TIME_CONSUMPTION_JSON = paths.PATH_TO_TIME_CONSUMPTION_JSON
model_utils.write_classifier_time_consumption_to_file(time_in_minutes_per_classifier,PATH_TO_TIME_CONSUMPTION_JSON)