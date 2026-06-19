import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

# Load dataset
file_path = "/content/DATASET - Sheet1.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Remove outliers in WATER REQUIREMENT
q1 = df['WATER REQUIREMENT'].quantile(0.25)
q3 = df['WATER REQUIREMENT'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['WATER REQUIREMENT'] >= lower_bound) & (df['WATER REQUIREMENT'] <= upper_bound)]

# One-hot encoding categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['CROP TYPE', 'SOIL TYPE', 'REGION', 'WEATHER CONDITION']])
encoded_feature_names = encoder.get_feature_names_out(['CROP TYPE', 'SOIL TYPE', 'REGION', 'WEATHER CONDITION'])
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Define features and target
X = encoded_df
y = df['WATER REQUIREMENT']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train a deep neural network
model = nn.Sequential(
    nn.Linear(X_scaled.shape[1], 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/100 - Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy().flatten()

r2 = r2_score(y, y_pred)
print(f"Model Accuracy (R² Score): {r2:.2f}")

# Save model and encoders
torch.save(model.state_dict(), "water_requirement_model.pth")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
