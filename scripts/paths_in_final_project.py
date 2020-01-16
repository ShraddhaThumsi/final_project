import os
DATA_ROOT = os.path.join("..",os.path.join("..","data"))
PATH_TO_DATAFRAME = os.path.join(DATA_ROOT,"grouped_features_Reach2019_matchinglabels.csv")
PATH_TO_GRAPHS_AND_SCORES = os.path.join("..",os.path.join("..","graphs_and_scores"))
PATH_TO_SCORES_JSON = os.path.join(PATH_TO_GRAPHS_AND_SCORES,"scores_per_classifier.json")
PATH_TO_TIME_CONSUMPTION_JSON = os.path.join(PATH_TO_GRAPHS_AND_SCORES,"time_consumption_per_classifier.json")