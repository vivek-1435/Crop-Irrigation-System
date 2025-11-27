import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

# Train a deep neural network
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)

# Evaluate model
y_pred = model.predict(X_scaled).flatten()
r2 = r2_score(y, y_pred)
print(f"Model Accuracy (R² Score): {r2:.2f}")

# Save model and encoders
model.save("water_requirement_model.h5")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
