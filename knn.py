import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
import pickle

# Load the dataset
data = pd.read_csv('preprocessed_final.csv')

# Separate features and labels
X = data.drop('correct', axis=1)
y = data['correct']

# Preprocess the data by scaling numerical features and one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['session_id', 'question_number', 'room_coor_y_mean', 'room_coor_x_mean', 'screen_coor_x_mean', 'elapsed_time_sum', 'screen_coor_y_mean', 'level_mean', 'music_max', 'name_nunique', 'room_fqid_nunique', 'event_name_nunique', 'fqid_count']),
        ('cat', OneHotEncoder(), ['level_group'])
    ])

X_processed = preprocessor.fit_transform(X)

# Create the kNN model
k = 5  # You can choose the number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Perform k-fold cross-validation
k_fold = 10  # Choose the number of folds
kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(X_processed)):
    X_train, X_test = X_processed[train_index], X_processed[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    f1_score = cross_val_score(knn, X_test, y_test, scoring='f1_macro').mean()

    print(f'Fold {i + 1} - Accuracy: {accuracy:.2f}, F1-score: {f1_score:.2f}')

    # Save the kNN model
    model_output_path = f'./knn_models/knn_fold_{i + 1}.pkl'
    with open(model_output_path, 'wb') as f:
        pickle.dump(knn, f)

# Calculate average accuracy and F1-score across all folds
accuracy_scores = cross_val_score(knn, X_processed, y, cv=kf, scoring='accuracy')
f1_scores = cross_val_score(knn, X_processed, y, cv=kf, scoring='f1_macro')

avg_accuracy = accuracy_scores.mean()
avg_f1_score = f1_scores.mean()
print(f'\nAverage accuracy across {k_fold}-fold cross-validation: {avg_accuracy:.2f}')
print(f'Average F1-score across {k_fold}-fold cross-validation: {avg_f1_score:.2f}')