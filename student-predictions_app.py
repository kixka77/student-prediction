import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Title
st.title("Student Performance & Study Habits Predictor")

# Input fields
st.subheader("Enter Your Study Habits & Academic Information")

hours_studied = st.slider("Hours Studied Per Day", 0, 12, 4)
attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
sleep_hours = st.slider("Hours of Sleep", 0, 12, 6)
assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 75)
extra_activities = st.selectbox("Participates in Extra Activities?", ["Yes", "No"])
past_grades = st.slider("Average Past Grades (%)", 0, 100, 85)

# Convert categorical input
extra_activities_val = 1 if extra_activities == "Yes" else 0

# Assemble input
user_input = np.array([[hours_studied, attendance, sleep_hours,
                        assignments_completed, extra_activities_val, past_grades]])

# Create mock dataset (for development/testing)
np.random.seed(42)
X_mock = np.random.randint(0, 100, (200, 6))
y_mock = np.where(np.mean(X_mock, axis=1) > 60, 1, 0)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mock)
user_scaled = scaler.transform(user_input)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_mock, test_size=0.2, random_state=42)

# DNN model (with fixed input)
model_dnn = Sequential([
    Input(shape=(X_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

# Predictions
y_pred = (model_dnn.predict(X_test) > 0.5).astype(int)
user_pred = (model_dnn.predict(user_scaled) > 0.5).astype(int)

# Show evaluation metrics
st.subheader("Model Evaluation Metrics")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

# Display result
st.subheader("Prediction Result")

if user_pred == 1:
    result = "Satisfactory / Excellent Performance"
else:
    if past_grades < 50 or hours_studied < 2:
        result = "At Risk"
    else:
        result = "Needs Improvement"

st.write(f"**Prediction:** {result}")
