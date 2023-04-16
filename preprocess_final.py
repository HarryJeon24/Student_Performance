from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
train_data = pd.read_csv('preprocessed.csv')

data = train_data[['session_id','level_group','room_coor_y_mean', 'room_coor_x_mean', 'screen_coor_x_mean', 'elapsed_time_sum', 'screen_coor_y_mean', 'level_mean', 'music_max', 'name_nunique','room_fqid_nunique', 'event_name_nunique','fqid_count', 'question_number', 'correct']]

# Save the data as CSV
filename = "preprocessed_final.csv"
data.to_csv(filename, index=False)

# Load the data
data = pd.read_csv('preprocessed_final.csv')

# Specify the columns you want to use for training and testing
features = ['room_coor_y_mean', 'room_coor_x_mean', 'screen_coor_x_mean', 'elapsed_time_sum', 'screen_coor_y_mean', 'level_mean', 'music_max', 'name_nunique','room_fqid_nunique', 'event_name_nunique','fqid_count']
target = 'correct'

# Group the data by session_id and question_number
grouped_data = data.groupby(['session_id', 'question_number'])

# Initialize empty dataframes for train and test data
train_data = pd.DataFrame(columns=data.columns)
test_data = pd.DataFrame(columns=data.columns)

# For each group, predict the correct column based on previous information within the group
for _, group in grouped_data:
    if len(group) > 1:
        # Split the data within the group into train and test
        group_train_data, group_test_data = train_test_split(group, test_size=0.2, random_state=42)

        # Append the split data to the respective train and test dataframes using pandas.concat
        train_data = pd.concat([train_data, group_train_data])
        test_data = pd.concat([test_data, group_test_data])
    else:
        # If there is only one sample in the group, add it to the train set using pandas.concat
        train_data = pd.concat([train_data, group])

# Split the data into features and labels
train_features = train_data[features]
train_labels = train_data[target]
test_features = test_data[features]
test_labels = test_data[target]

# Save the data as CSV
filename = "train_features.csv"
train_features.to_csv(filename, index=False)

filename = "train_labels.csv"
train_labels.to_csv(filename, index=False)

filename = "test_features.csv"
test_features.to_csv(filename, index=False)

filename = "test_labels.csv"
test_labels.to_csv(filename, index=False)