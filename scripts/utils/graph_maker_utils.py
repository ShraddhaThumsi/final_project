import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker

def get_scores_as_list(metrics_per_classifier_dict):
    '''
    This function takes the scores generated by each classifier into a list.
    :param metrics_per_classifier_dict:
    :return:
    '''
    overall_scores = list()
    for classifier_name in metrics_per_classifier_dict.keys():
        current_score_dict = metrics_per_classifier_dict[classifier_name]

        f1 = current_score_dict["f1"]
        precision = current_score_dict["precision"]
        recall = current_score_dict["recall"]
        accuracy = current_score_dict["accuracy"]

        overall_scores.append((classifier_name,float(precision),float(recall),float(f1),float(accuracy)))
    overall_scores.sort(key=lambda tup:tup[3])
    return overall_scores

def plot_scores(metrics_per_model_dict, path_to_scores_graph):
    '''
    This function takes the list of metrics per each classifier generated from get_scores_as_list and then plot the 
    scores in a bar chart
    :param metrics_per_model_dict: 
    :param path_to_scores_graph: 
    :return: 
    ''''''
    :param metrics_per_model_dict: 
    :param path_to_scores_graph: 
    :return: 
    '''
    over_all_score_list = get_scores_as_list(metrics_per_model_dict)
    (classifier_names,precision_scores,recall_scores,f1_scores,accuracy_scores) = map(list, zip(*over_all_score_list))
    x_vals = np.array(list(range(len(classifier_names))))
    plt.figure(figsize=(10, 9))
    #color palette courtesy for the bar graph: colorhunt.com
    rects1 = plt.bar(x_vals - 0.2, precision_scores, width=0.2, color="#F4B0C7", align='center', label="Precision")
    rects2 = plt.bar(x_vals, recall_scores, width=0.2, color="#AC8BAF", align='center', label="Recall")
    rects3 = plt.bar(x_vals + 0.2, f1_scores, width=0.2, color="#FFD800", align='center', label="F1")
    rects4 = plt.bar(x_vals + 0.4, accuracy_scores, width=0.2, color="#6FB98F", align='center', label="Accuracy")
    plt.xticks(x_vals, classifier_names)


    def autolabel(rects, scores):
        """
        Attach a text label above each bar displaying its height
        """
        max_score = max(scores)
        for rect, score in zip(rects, scores):
            height = rect.get_height()
            color = "red" if score == max_score else "black"
            font_dict = {"weight":"normal",
                         "color":color,
                         "size":6}
            plt.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                     "{:.3f}".format(float(score)),
                     ha='center', va='bottom', fontdict=font_dict)

    # Label the height of all bars with actual scores
    autolabel(rects1, precision_scores)
    autolabel(rects2, recall_scores)
    autolabel(rects3, f1_scores)
    autolabel(rects4, accuracy_scores)

    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    plt.title("Classifier score comparison")
    plt.xlabel("Model name")
    plt.ylabel("%-score")
    plt.subplots_adjust(left=0.075, right=.925, bottom=.2, top=.95)
    plt.savefig(path_to_scores_graph)
    plt.show()


def plot_f1_vs_time(time_dict,f1_dict,path_to_graph):
    tup_list = []

    for model_name in time_dict.keys():
        time_taken_by_model = float(time_dict[model_name])
        f1_from_model = float(f1_dict[model_name]["f1"])
        tup_list.append((time_taken_by_model,f1_from_model))
    tup_list.sort(key=lambda t:t[0])
    model_names_sorted_by_time = []

    (time_list,f1_list) = zip(*tup_list)
    for time in list(time_list):
        for model_name,unsorted_time in time_dict.items():
            if float(unsorted_time) == time:
                model_names_sorted_by_time.append(model_name)
    plt.figure(figsize=(20,10))
    color_list = ["#27bf0f","#cc7722","#005aff","#FF0000","#000080","#a500ff"]
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set(ylim=(0.,1.))
    for index,time in enumerate(time_list):

        f1_at_index = f1_list[index]
        font_dict = {"weight": "normal",
                     "color": color_list[index],
                     "size": 8}

        rect = plt.bar(time,f1_at_index,width=0.1,color=color_list[index])
        height = rect[0].get_height()
        plt.text(rect[0].get_x() + rect[0].get_width() / 2., 1.00 * height,
                 model_names_sorted_by_time[index],
                 ha='center', va='bottom', fontdict=font_dict,rotation=90)



    plt.xlabel("Time taken in minutes")
    plt.ylabel("% F1 score")
    plt.title(f"Time v/s F1 score trade off in {len(time_dict)} classification models")
    plt.savefig(path_to_graph)
    plt.show()

def read_classifier_dicts_from_json(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    return data