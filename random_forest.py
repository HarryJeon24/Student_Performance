import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

# Load train_features.csv and train_labels.csv
train_features = pd.read_csv("final_train_data/train_features.csv")
train_labels = pd.read_csv("final_train_data/train_labels.csv")

# Perform KFold split with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store F1 score and accuracy for each model
f1_scores = []
accuracies = []

# Loop through each fold
for train_index, test_index in kf.split(train_features):
    print("Start training!")
    # Split the data into train and test sets
    X_train, X_test = train_features.iloc[train_index], train_features.iloc[test_index]
    y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]

    print("Finish split!")
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf.predict(X_test)

    # Calculate F1 score and accuracy
    f1 = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy = ", accuracy, " F1 = ", f1)
    # Append F1 score and accuracy to lists
    f1_scores.append(f1)
    accuracies.append(accuracy)

# Print F1 score and accuracy for each model
for i in range(len(f1_scores)):
    print(f"Model {i + 1}:")
    print(f"F1 Score: {f1_scores[i]:.4f}")
    print(f"Accuracy: {accuracies[i]:.4f}")
    print("--------------")
