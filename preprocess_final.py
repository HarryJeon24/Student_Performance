import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
train_data = pd.read_csv('preprocessed.csv')

data = train_data[['session_id','level_group','room_coor_y_mean', 'room_coor_x_mean', 'screen_coor_x_mean', 'elapsed_time_sum', 'screen_coor_y_mean', 'level_mean', 'music_max', 'name_nunique','room_fqid_nunique', 'event_name_nunique','fqid_count', 'question_number', 'correct']]

# Save the data as CSV
filename = "preprocessed_final.csv"
data.to_csv(filename, index=False)