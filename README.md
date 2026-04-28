# Air Writing Recognition System (Arduino + MPU6050 + ML)

An intelligent system that recognizes letters written in the air using motion sensors and machine learning.

This project uses an MPU6050 (accelerometer + gyroscope) with Arduino to capture hand gestures and a Python-based ML model to classify them in real time.

---

## Features

-  Air gesture recognition (W, O, L)
-  Real-time data capture using Arduino
-  Machine Learning (Random Forest – no TensorFlow required)
-  High accuracy using 6-axis IMU data
-  Fast and lightweight prediction
-  Easily extendable to more gestures

---

## How It Works

1. User writes a letter in the air  
2. MPU6050 captures motion data (ax, ay, az, gx, gy, gz)  
3. Arduino sends data via Serial communication  
4. Python script:
   - Normalizes the data  
   - Extracts motion features  
   - Feeds into trained ML model  
5. Model predicts the gesture in real time  

---

## Tech Stack

**Hardware:**
- Arduino
- MPU6050 Sensor

**Software:**
- Python

**Libraries Used:**
- NumPy  
- scikit-learn  
- PySerial  
- Joblib  

---

## Project Structure
