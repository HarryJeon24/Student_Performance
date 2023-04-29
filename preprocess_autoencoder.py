import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Read the CSV file
df = pd.read_csv('train.csv')

# Preprocess the data (handle non-numeric and missing values)
# If you have non-numeric columns, convert them to numeric values
df['event_name'] = pd.Categorical(df['event_name']).codes
df['name'] = pd.Categorical(df['name']).codes
df['text'] = pd.Categorical(df['text']).codes
df['fqid'] = pd.Categorical(df['fqid']).codes
df['room_fqid'] = pd.Categorical(df['room_fqid']).codes
df['text_fqid'] = pd.Categorical(df['text_fqid']).codes

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

# Create a PyTorch DataLoader
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Set parameters and create the autoencoder
input_dim = X_normalized.shape[1]
encoding_dim = 10
model = Autoencoder(input_dim, encoding_dim)

# Set the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extract features using the trained autoencoder
with torch.no_grad():
    X_encoded = model.encoder(torch.tensor(X_normalized, dtype=torch.float32)).numpy()

print("Original data shape:", X_normalized.shape)
print("Encoded data shape:", X_encoded.shape)

# Save the encoded data to a CSV file
encoded_df = pd.DataFrame(X_encoded, columns=[f'encoded_feature{i+1}' for i in range(encoding_dim)])
encoded_df.to_csv('encoded_data.csv', index=False)