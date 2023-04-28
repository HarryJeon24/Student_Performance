from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
train_data = pd.read_csv('preprocessed/preprocessed.csv')

data = train_data[
    ['session_id', 'question_number', 'level_group', 'room_coor_y_mean', 'room_coor_x_mean', 'screen_coor_x_mean',
     'elapsed_time_sum', 'screen_coor_y_mean', 'level_mean', 'music_max', 'name_nunique', 'room_fqid_nunique',
     'event_name_nunique', 'fqid_count', 'correct']]

# Save the data as CSV
filename = "preprocessed/preprocessed_final.csv"
data.to_csv(filename, index=False)

# Load the data
data = pd.read_csv('preprocessed/preprocessed_final.csv')

# Specify the columns you want to use for training and testing
features = ['question_number', 'room_coor_y_mean', 'room_coor_x_mean', 'screen_coor_x_mean', 'elapsed_time_sum', 'screen_coor_y_mean',
            'level_mean', 'music_max', 'name_nunique', 'room_fqid_nunique', 'event_name_nunique', 'fqid_count']
target = 'correct'

# Group the data by session_id
grouped_data = data.groupby('session_id')

train_features = train_data[features]
train_labels = train_data[target]

# Save the data as CSV
filename = "final_train_data/train_features.csv"
train_features.to_csv(filename, index=False)

filename = "final_train_data/train_labels.csv"
train_labels.to_csv(filename, index=False)