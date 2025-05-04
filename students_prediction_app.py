import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Title
st.title("Academic Performance & Study Habits Predictor")

# User Inputs
hours_studied = st.slider("Hours Studied per Day", 0.0, 12.0, 4.0)
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
assignments_completed = st.slider("Assignments Completed (out of 10)", 0, 10, 7)
extra_activities = st.slider("No. of Extra Activities", 0, 5, 2)
past_grades = st.slider("Past Grade (%)", 0.0, 100.0, 75.0)

# Define real classification logic based on thresholds
def assign_category(row):
    score = (
        (row["hours_studied"] * 0.2) +
        (row["attendance"] * 0.2) +
        (row["sleep_hours"] * 0.1) +
        (row["assignments_completed"] * 0.2) +
        (row["past_grades"] * 0.2) -
        (row["extra_activities"] * 0.1)
    )
    if score < 50:
        return "At Risk"
    elif score < 65:
        return "Needs Improvement"
    elif score < 80:
        return "Satisfactory"
    else:
        return "Excellent"

# Generate realistic dataset
np.random.seed(42)
data = {
    "hours_studied": np.random.normal(5, 2, 200).clip(0, 12),
    "attendance": np.random.normal(85, 10, 200).clip(50, 100),
    "sleep_hours": np.random.normal(7, 1.5, 200).clip(3, 10),
    "assignments_completed": np.random.randint(0, 11, 200),
    "extra_activities": np.random.randint(0, 6, 200),
    "past_grades": np.random.normal(75, 15, 200).clip(0, 100),
}
df = pd.DataFrame(data)
df["performance"] = df.apply(assign_category, axis=1)

# Features and Labels
X = df.drop("performance", axis=1)
y = df["performance"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Gradient Boosting Model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# DNN Model
model_dnn = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Encode labels numerically for DNN
label_map = {label: idx for idx, label in enumerate(np.unique(y))}
y_train_num = np.array([label_map[label] for label in y_train])
model_dnn.fit(X_train, y_train_num, epochs=20, batch_size=16, verbose=0)

# Combine predictions (majority logic)
input_data = np.array([[hours_studied, attendance, sleep_hours, assignments_completed, extra_activities, past_grades]])
input_scaled = scaler.transform(input_data)

gb_pred = gb_model.predict(input_scaled)[0]
dnn_pred_idx = np.argmax(model_dnn.predict(input_scaled, verbose=0))
dnn_pred = list(label_map.keys())[list(label_map.values()).index(dnn_pred_idx)]

final_prediction = gb_pred if gb_pred == dnn_pred else dnn_pred

# Metrics
y_pred_gb = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_gb)
precision = precision_score(y_test, y_pred_gb, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_gb, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_gb, average='weighted', zero_division=0)

# Results
st.subheader("Prediction")
st.success(f"Predicted Performance: **{final_prediction}**")

st.subheader("Model Evaluation Metrics (Gradient Boosting)")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")
