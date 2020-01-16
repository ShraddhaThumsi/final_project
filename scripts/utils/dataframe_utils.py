import pandas
import numpy


def get_dataframe_from_csv(path_to_csv):
    return pandas.read_csv(path_to_csv)


def get_balanced_dataframe(dataframe,amt_per_pos):
    numpy.random.seed(13)
    grouping_key = "PMCID"
    grouped_by_paper = dataframe.groupby(grouping_key)
    grouped_by_paper = list(grouped_by_paper)
    all_rows = None
    for idx,paper_df in grouped_by_paper:

        positive_rows = paper_df[paper_df["label"]==True]
        negative_rows = paper_df[paper_df["label"]==False]

        if all_rows is None:
            all_rows = sample_rows_from_paper(positive_rows.values,negative_rows.values,amt_per_pos)

        else:
            chosen_rows = sample_rows_from_paper(positive_rows.values,negative_rows.values,amt_per_pos)
            all_rows.extend(chosen_rows)

    return pandas.DataFrame(all_rows, columns=dataframe.columns)


def sample_rows_from_paper(all_positive_rows, all_negative_rows, requested_num_of_negs_per_pos):
    """
    this function is an aide to the main balancer function. this simply takes the feature values and the number of
    negatives per positive needed per paper, samples them, and returns a collective list of event-context feature values.
    :param all_positive_rows: all the positive event-context annotations and the feature values obtained from Reach 2019
    :param all_negative_rows: all the negative event-context annotations and the feature values obtained from Reach 2019
    :param requested_num_of_negs_per_pos: number of negative samples per positive sample that we need to pick.
    :return: A list of event-context data points, having both positive examples and negative examples.
    """
    num_of_available_negs = len(all_negative_rows)
    num_of_available_pos = len(all_positive_rows)
    if num_of_available_negs < num_of_available_pos:
        num_of_pos_to_sample = num_of_available_negs * requested_num_of_negs_per_pos
        if num_of_pos_to_sample > num_of_available_pos:
            raise ValueError("Requested number of positive examples are more than available samples")
        numpy.random.shuffle(all_positive_rows)
        chosen_positive_rows = all_positive_rows[:num_of_pos_to_sample]
        all_chosen_rows = []
        all_chosen_rows.extend(all_negative_rows)
        all_chosen_rows.extend(chosen_positive_rows)
    else:
        num_of_negs_to_sample = num_of_available_pos * requested_num_of_negs_per_pos
        if num_of_negs_to_sample > num_of_available_negs:
            raise ValueError("Requested number of negative samples are more than available samples")
        numpy.random.shuffle(all_negative_rows)
        chosen_negative_rows = all_negative_rows[:num_of_negs_to_sample]
        all_chosen_rows = []
        all_chosen_rows.extend(all_positive_rows)
        all_chosen_rows.extend(chosen_negative_rows)
    return all_chosen_rows