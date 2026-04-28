import serial
import time
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

# Settings
PORT       = 'COM5'         
BAUD       = 9600
N_SAMPLES  = 30              
TARGET_STEPS = 128           
FEATURES   = 3
MODEL_PATH = "model/gesture_model.h5"
CLASSES    = ["W", "O", "L"]
THRESHOLD  = 0.80            


print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except OSError:
    print(f"\n[ERROR] Could not find '{MODEL_PATH}'!")
    exit(1)
print("Model loaded.\n")

try:
    print(f"Connecting to {PORT}...")
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2)
    ser.reset_input_buffer()
    print("Connected!\n")
except serial.SerialException:
    print(f"\n[ERROR] Could not connect to {PORT}. Is the Serial Monitor closed?")
    exit(1)

def read_gesture():
    """Collect N_SAMPLES readings from Arduino."""
    ser.reset_input_buffer()
    ser.readline() # clear partial line
    data = []
    while len(data) < N_SAMPLES:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                row = [float(parts[0]), float(parts[1]), float(parts[2])]
                data.append(row)
            except ValueError:
                pass
    return np.array(data, dtype=np.float32)

def preprocess(raw):
    """Interpolates to 128 steps, then normalizes."""
    # 1. Stretch from 30 samples to 128 samples
    curr_steps = raw.shape[0]
    x_old = np.linspace(0, 1, curr_steps)
    x_new = np.linspace(0, 1, TARGET_STEPS)
    
    resampled = np.zeros((TARGET_STEPS, FEATURES))
    for i in range(FEATURES):
        f = interp1d(x_old, raw[:, i], kind='linear')
        resampled[:, i] = f(x_new)
        
    # 2. Z-score normalize
    mean = resampled.mean(axis=0)
    std  = resampled.std(axis=0) + 1e-8
    return (resampled - mean) / std

def predict(arr):
    x = arr[np.newaxis, ...]                    # shape becomes (1, 128, 3)
    probs = model.predict(x, verbose=0)[0]      
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs

print("Press ENTER to record a gesture (write W, O, or L in the air).")
print("Press Ctrl+C to quit.\n")

try:
    while True:
        input("Press ENTER then write your letter...")
        print(f"  Recording {N_SAMPLES} samples...")

        raw   = read_gesture()
        arr   = preprocess(raw) # Now outputs (128, 3)
        label, conf, probs = predict(arr)

        prob_str = "  ".join(f"{c}={probs[i]*100:.0f}%" for i, c in enumerate(CLASSES))
        print(f"  [{prob_str}]")

        if conf >= THRESHOLD:
            print(f"  >>> {label}  (confidence: {conf*100:.0f}%)\n")
        else:
            print(f"  >>> Unclear  (best guess: {label} at {conf*100:.0f}%)\n")

except KeyboardInterrupt:
    print("\nStopped.")
    ser.close()