import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("Hybrid Student Performance and Study Habit Predictor")

# Input form
with st.form("input_form"):
    st.subheader("Enter your study habits and academic behavior:")
    hours_studied = st.slider("Hours Studied per Day", 0.0, 12.0, 1.0)
    sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 6.0)
    attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
    assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 80)
    quiz_avg = st.slider("Quiz Average (%)", 0, 100, 70)
    internet_usage = st.slider("Internet Usage for Study (Hours/day)", 0.0, 12.0, 2.0)
    submit = st.form_submit_button("Predict")

# Prepare real input as a DataFrame
real_input = pd.DataFrame([[
    hours_studied, sleep_hours, attendance,
    assignments_completed, quiz_avg, internet_usage
]], columns=[
    "Hours_Studied", "Sleep_Hours", "Attendance",
    "Assignments_Completed", "Quiz_Avg", "Internet_Usage"
])

# Example real training dataset (you can replace this with actual historical data)
data = pd.DataFrame({
    "Hours_Studied": np.random.uniform(0, 12, 200),
    "Sleep_Hours": np.random.uniform(4, 10, 200),
    "Attendance": np.random.randint(50, 100, 200),
    "Assignments_Completed": np.random.randint(50, 100, 200),
    "Quiz_Avg": np.random.randint(50, 100, 200),
    "Internet_Usage": np.random.uniform(0, 12, 200),
})

# Mock label creation based on score (adjust this to match your criteria)
total_score = (
    0.3 * data["Hours_Studied"] +
    0.1 * data["Sleep_Hours"] +
    0.2 * (data["Attendance"] / 100) +
    0.2 * (data["Assignments_Completed"] / 100) +
    0.2 * (data["Quiz_Avg"] / 100)
)
labels = pd.cut(
    total_score,
    bins=[-np.inf, 3.5, 5.0, 6.5, np.inf],
    labels=["At Risk", "Needs Improvement", "Satisfactory", "Excellent"]
)
data["Performance"] = labels

# Encode labels
le = LabelEncoder()
data["Performance_encoded"] = le.fit_transform(data["Performance"])

# Split and scale
X = data.drop(columns=["Performance", "Performance_encoded"])
y = data["Performance_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Clear previous sessions and train DNN
K.clear_session()
model_dnn = Sequential()
model_dnn.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model_dnn.add(Dense(32, activation='relu'))
model_dnn.add(Dense(len(np.unique(y)), activation='softmax'))
model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

# Real prediction logic
if submit:
    input_scaled = scaler.transform(real_input)

    pred_gb = gb_model.predict(input_scaled)
    pred_dnn = model_dnn.predict(input_scaled)
    pred_dnn_label = np.argmax(pred_dnn, axis=1)

    # Majority vote (simple averaging of models)
    final_pred = int(round((pred_gb[0] + pred_dnn_label[0]) / 2.0))
    final_label = le.inverse_transform([final_pred])[0]

    st.success(f"Predicted Performance: **{final_label}**")

    # Show evaluation metrics
    y_pred_test = gb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

    st.subheader("Model Evaluation (Gradient Boosting)")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
