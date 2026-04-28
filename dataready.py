import os
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras import layers, models


# 1. SETUP AND CONFIGURATION

folders = {
    "W": "C:/Users/PARTH GARG/Downloads/data/wing",
    "O": "C:/Users/PARTH GARG/Downloads/data/ring",
    "L": "C:/Users/PARTH GARG/Downloads/data/slope"
}

TARGET_STEPS = 128
FEATURES = 3 # Sticking to ax, ay, az
label_map = {"W": 0, "O": 1, "L": 2}


# 2. DATA PREPROCESSING

def load_and_resample(file_path):
    clean_data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3 or "-" in parts[0]: 
                continue
            try:
                clean_data.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
                
    raw_data = np.array(clean_data)
    if len(raw_data) < 10: 
        return None

    # Stretch/squeeze data to exactly 128 steps
    curr_steps = raw_data.shape[0]
    x_old = np.linspace(0, 1, curr_steps)
    x_new = np.linspace(0, 1, TARGET_STEPS)
    
    resampled_data = np.zeros((TARGET_STEPS, FEATURES))
    for i in range(FEATURES):
        f = interp1d(x_old, raw_data[:, i], kind='linear')
        resampled_data[:, i] = f(x_new)
    return resampled_data

X, y = [], []
for letter, path in folders.items():
    print(f"Loading data for {letter}...")
    for file in os.listdir(path):
        data = load_and_resample(os.path.join(path, file))
        if data is not None:
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8
            X.append((data - mean) / std)
            y.append(label_map[letter])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"\nDataset Ready! X shape: {X.shape}, y shape: {y.shape}")

if len(X) == 0:
    print("ERROR: Dataset is empty. Check your folder paths.")
    exit()


# 3. BUILDING & TRAINING THE MODEL

model = models.Sequential([
    layers.Input(shape=(TARGET_STEPS, FEATURES)),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- STARTING TRAINING ---")
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)
print("--- TRAINING COMPLETE ---")

# Save model in a 'model' directory so recognize.py can find it
os.makedirs("model", exist_ok=True)
model.save("model/gesture_model.h5")
print("Model saved to model/gesture_model.h5")