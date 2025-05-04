import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Title
st.title("Student Performance and Study Habits Predictor")

# Real-time input form
st.subheader("Enter Student Data")
study_time = st.slider("Study Time (hrs/day)", 0, 10, 2)
sleep_time = st.slider("Sleep Time (hrs/day)", 0, 10, 6)
attendance = st.slider("Attendance (%)", 0, 100, 85)
assignments = st.slider("Assignment Completion (%)", 0, 100, 80)
participation = st.slider("Class Participation (%)", 0, 100, 70)
social_media = st.slider("Social Media Usage (hrs/day)", 0, 10, 3)

# Example dataset (mockup for structure)
data = {
    "study_time": np.random.randint(1, 6, 200),
    "sleep_time": np.random.randint(4, 10, 200),
    "attendance": np.random.randint(50, 100, 200),
    "assignments": np.random.randint(50, 100, 200),
    "participation": np.random.randint(30, 100, 200),
    "social_media": np.random.randint(0, 6, 200),
}

df = pd.DataFrame(data)

# Target based on custom logic (for demonstration)
conditions = (
    (df["study_time"] >= 4) & 
    (df["attendance"] >= 85) & 
    (df["assignments"] >= 85) &
    (df["participation"] >= 80) &
    (df["social_media"] <= 3)
)

df["performance"] = np.where(conditions, "Excellent", 
                     np.where((df["attendance"] < 60) | (df["assignments"] < 60), "At Risk",
                     np.where((df["attendance"] < 75) | (df["assignments"] < 75), "Needs Improvement", 
                     "Satisfactory")))

# Feature and label split
X = df.drop("performance", axis=1)
y = df["performance"]

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# DNN model
model_dnn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])
model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

# Prepare user input
input_data = np.array([[study_time, sleep_time, attendance, assignments, participation, social_media]])
input_scaled = scaler.transform(input_data)

# Predictions
gb_pred_idx = gb_model.predict(input_scaled)[0]
dnn_pred_idx = np.argmax(model_dnn.predict(input_scaled, verbose=0))

gb_pred = label_encoder.inverse_transform([gb_pred_idx])[0]
dnn_pred = label_encoder.inverse_transform([dnn_pred_idx])[0]

# Display Results
st.subheader("Predicted Academic Performance:")
st.write(f"**Gradient Boosting:** {gb_pred}")
st.write(f"**Deep Neural Network:** {dnn_pred}")

# Evaluate
gb_test_pred = gb_model.predict(X_test)
dnn_test_pred = np.argmax(model_dnn.predict(X_test, verbose=0), axis=1)

st.subheader("Model Evaluation Metrics:")
st.write("**Gradient Boosting Classifier:**")
st.write(f"- Accuracy: {accuracy_score(y_test, gb_test_pred):.2f}")
st.write(f"- Precision: {precision_score(y_test, gb_test_pred, average='weighted'):.2f}")
st.write(f"- Recall: {recall_score(y_test, gb_test_pred, average='weighted'):.2f}")
st.write(f"- F1 Score: {f1_score(y_test, gb_test_pred, average='weighted'):.2f}")

st.write("**Deep Neural Network:**")
st.write(f"- Accuracy: {accuracy_score(y_test, dnn_test_pred):.2f}")
st.write(f"- Precision: {precision_score(y_test, dnn_test_pred, average='weighted'):.2f}")
st.write(f"- Recall: {recall_score(y_test, dnn_test_pred, average='weighted'):.2f}")
st.write(f"- F1 Score: {f1_score(y_test, dnn_test_pred, average='weighted'):.2f}")
