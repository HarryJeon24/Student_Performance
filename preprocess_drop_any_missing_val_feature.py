import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
train_data = pd.read_csv('train.csv')
train_labels = pd.read_csv('train_labels.csv')

# Drop the specified columns
train_data = train_data.drop(columns=['page', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration', 'text', 'fqid', 'text_fqid', 'fullscreen', 'hq', 'music'])

# Extract the question numbers from the session_id column in the train_labels.csv file
train_labels['question_number'] = train_labels['session_id'].str.extract(r'q(\d+)$').astype(np.int64)

# Convert the session_id column in the train_labels.csv file to integer values
train_labels['session_id'] = train_labels['session_id'].str.extract(r'(\d+)_q\d+$').astype(np.int64)

agg_funcs = {}
for col in train_data.columns:
    if np.issubdtype(train_data[col].dtype, np.number):  # Updated to use np.issubdtype
        agg_funcs[col] = ['mean', 'sum', 'min', 'max']
    else:
        agg_funcs[col] = ['first', 'last', 'count', 'nunique']

# Aggregate the game logs by session_id
train_data_agg = train_data.groupby('session_id').agg(agg_funcs)

# Flatten the column index
train_data_agg.columns = ['_'.join(col) for col in train_data_agg.columns.values]

# Merge the aggregated game logs with the labels
data = pd.merge(train_data_agg.reset_index(), train_labels, on='session_id')

# Save the data as CSV
filename = "preprocessed1.csv"
data.to_csv(filename, index=False)

# Select only numeric columns
data_numeric = data.select_dtypes(include=np.number)

# Calculate the correlation matrix
corr_matrix = data.corr(numeric_only=True)  # Explicitly specify numeric_only=True

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Reds)
plt.show()

# Handle missing values
data_numeric = data_numeric.dropna(axis=1)

# Save the data as CSV
# filename = "preprocessed_numeric.csv"
# data_numeric.to_csv(filename, index=False)

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