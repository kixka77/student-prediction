import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Streamlit UI
st.title("Student Performance & Study Habit Predictor")

st.markdown("### Enter your academic and study habit indicators:")

# Input fields
study_hours = st.number_input("Study Hours Per Week", 0, 80, 10)
attendance = st.slider("Class Attendance Rate (%)", 0, 100, 85)
sleep_hours = st.slider("Sleep Hours Per Night", 0, 12, 7)
extracurricular = st.slider("Extracurricular Involvement (0=None, 10=High)", 0, 10, 5)
assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 90)
motivation_level = st.slider("Self-Motivation (0=None, 10=High)", 0, 10, 7)

input_features = np.array([[study_hours, attendance, sleep_hours, extracurricular, assignments_completed, motivation_level]])

# Simulate training data (temporarily for model initialization)
X_mock = np.random.randint(0, 100, size=(200, 6))
y_mock = np.random.randint(0, 4, size=(200,))  # 0 = At Risk, 1 = Needs Improvement, 2 = Satisfactory, 3 = Excellent

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mock)
X_input = scaler.transform(input_features)

# Train Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_scaled, y_mock)

# Train DNN
model_dnn = Sequential([
    Dense(64, input_shape=(6,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_scaled, y_mock, epochs=20, batch_size=16, verbose=0)

# Hybrid prediction
gb_pred = gb_model.predict(X_input)[0]
dnn_pred = np.argmax(model_dnn.predict(X_input), axis=1)[0]
final_prediction = round((gb_pred + dnn_pred) / 2)

# Label mapping
labels = {0: "At Risk", 1: "Needs Improvement", 2: "Satisfactory", 3: "Excellent"}
st.subheader("Prediction:")
st.success(f"Your predicted performance is: **{labels[final_prediction]}**")

# Show evaluation metrics on mock data (optional)
if st.checkbox("Show Model Evaluation (Mock Training Data)"):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_mock, test_size=0.2, random_state=42)
    y_pred_gb = gb_model.predict(X_test)
    y_pred_dnn = np.argmax(model_dnn.predict(X_test), axis=1)

    st.write("### Gradient Boosting Metrics:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred_gb, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred_gb, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred_gb, average='weighted'):.2f}")

    st.write("### DNN Metrics:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_dnn):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred_dnn, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred_dnn, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred_dnn, average='weighted'):.2f}")
