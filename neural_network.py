import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Read the CSV file
df = pd.read_csv('preprocessed_final.csv')

# Preprocess the data (handle non-numeric and missing values)
df = df.apply(pd.to_numeric, errors='coerce')
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df.drop('correct', axis=1))
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(df['correct'].values, dtype=torch.long)

# Create the k-fold cross-validator
k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=42)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(train_dataloader, val_dataloader):
    # Set parameters and create the neural network
    input_dim = X_normalized.shape[1]
    hidden_dim = 128
    output_dim = 2  # Assuming binary classification (correct or not)
    model = MLP(input_dim, hidden_dim, output_dim)

    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the neural network
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f'/local/scratch/hjeon62/mlp_output/mlp_model_fold_{fold + 1}.pt')
    return model

# Train and evaluate the model with k-fold cross-validation
fold_accuracies = []
fold_f1_scores = []
for fold, (train_indices, val_indices) in enumerate(kfold.split(X_tensor, y_tensor)):
    X_train, X_val = X_tensor[train_indices], X_tensor[val_indices]
    y_train, y_val = y_tensor[train_indices], y_tensor[val_indices]

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = train_model(train_dataloader, val_dataloader)

    # Evaluate the model on the validation set
    model.eval()
    val_outputs = []
    val_targets = []
    with torch.no_grad():
        for data, targets in val_dataloader:
            outputs = model(data)
            val_outputs.append(outputs)
            val_targets.append(targets)

    val_outputs = torch.cat(val_outputs, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    val_predictions = torch.argmax(val_outputs, dim=1)
    val_accuracy = accuracy_score(val_targets, val_predictions)
    val_f1 = f1_score(val_targets, val_predictions)
    fold_accuracies.append(val_accuracy)
    fold_f1_scores.append(val_f1)
    print(f'Fold {fold + 1}/{k}, Validation accuracy: {val_accuracy:.4f}, Validation F1-score: {val_f1:.4f}')

# Calculate and print the average validation accuracy and F1-score across all folds
average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
average_f1_score = sum(fold_f1_scores) / len(fold_f1_scores)
print(f'Average validation accuracy across {k}-fold cross-validation: {average_accuracy:.4f}')
print(f'Average validation F1-score across {k}-fold cross-validation: {average_f1_score:.4f}')