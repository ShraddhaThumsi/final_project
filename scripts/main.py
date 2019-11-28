import utils.dataframe_utils as df_utils
import utils.model_utils as model_utils
import paths_in_final_project as paths

PATH_T0_DATAFRAME = paths.PATH_TO_DATAFRAME
print(PATH_T0_DATAFRAME)
dataframe = df_utils.get_dataframe_from_csv(PATH_T0_DATAFRAME)
print(dataframe.shape)
