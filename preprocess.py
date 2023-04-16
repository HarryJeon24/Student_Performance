import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to map question_number to level_group
def map_question_number_to_level_group(question_number):
    if 1 <= question_number <= 5:
        return '0-4'
    elif 6 <= question_number <= 13:
        return '5-12'
    elif 14 <= question_number <= 18:
        return '13-22'
    else:
        return 'unknown'

# Load the data
train_data = pd.read_csv('train.csv')
train_labels = pd.read_csv('train_labels.csv')

# Drop the columns with not sufficient numbers of values
train_data = train_data.drop(columns=['index', 'page', 'hover_duration', 'text', 'text_fqid'])

# Extract the question numbers from the session_id column in the train_labels.csv file
train_labels['question_number'] = train_labels['session_id'].str.extract(r'q(\d+)$').astype(np.int64)

# Convert the session_id column in the train_labels.csv file to integer values
train_labels['session_id'] = train_labels['session_id'].str.extract(r'(\d+)_q\d+$').astype(np.int64)

# Add 'level_group' column to train_labels based on 'question_number' column
train_labels['level_group'] = train_labels['question_number'].apply(map_question_number_to_level_group)

# Define the aggregation functions for all columns
agg_funcs = {}
for col in train_data.columns:
    if np.issubdtype(train_data[col].dtype, np.number):  # Updated to use np.issubdtype
        agg_funcs[col] = ['mean', 'sum', 'min', 'max']
    else:
        agg_funcs[col] = ['first', 'last', 'count', 'nunique']

# Aggregate the game logs by session_id and level_group
train_data_agg = train_data.groupby(['session_id', 'level_group']).agg(agg_funcs)

# Flatten the column index
train_data_agg.columns = ['_'.join(col) for col in train_data_agg.columns]

# Merge the aggregated game logs with the labels
data = pd.merge(train_data_agg.reset_index(), train_labels, on=['session_id', 'level_group'])

# Drop the specified columns
data = data.drop(columns=['session_id_mean', 'session_id_sum', 'session_id_min', 'session_id_max', 'level_group_first', 'level_group_last', 'level_group_count', 'level_group_nunique'])

# Save the data as CSV
filename = "preprocessed.csv"
data.to_csv(filename, index=False)

# Select only numeric columns
data_numeric = data.select_dtypes(include=np.number)

# Drop the specified columns
data = data.drop(columns=['session_id', 'level_group', 'question_number'])

# Calculate the correlation matrix
corr_matrix = data.corr(numeric_only=True)  # Explicitly specify numeric_only=True

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# Handle missing values
data_numeric = data_numeric.dropna(axis=1)

# Save the data as CSV
filename = "preprocessed_numeric.csv"
data_numeric.to_csv(filename, index=False)

# Calculate the mutual information
mutual_info = mutual_info_classif(data_numeric.drop(columns=['session_id', 'correct']), data['correct'])

# Create a dataframe with the feature importances
feature_importances = pd.DataFrame(mutual_info, index=data_numeric.drop(columns=['session_id', 'correct']).columns, columns=['importance'])  # Use data_numeric instead of data

# Sort the feature importances in descending order
feature_importances = feature_importances.sort_values(by='importance', ascending=False)


# Plot the feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
plt.xticks(rotation=90)
plt.show()