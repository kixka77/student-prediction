import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Simulated dataset (mock)
data = pd.DataFrame({
    'study_hours': np.random.randint(1, 10, 100),
    'sleep_hours': np.random.randint(4, 9, 100),
    'class_participation': np.random.randint(1, 10, 100),
    'assignments_done': np.random.randint(1, 10, 100),
    'quizzes_score': np.random.randint(1, 10, 100),
    'performance': np.random.choice(['Needs Improvement', 'Satisfactory', 'Excellent', 'At Risk'], 100)
})

X = data.drop('performance', axis=1)
y = data['performance']

# Encode target
y = pd.factorize(y)[0]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- GBDT Model
gbdt_model = GradientBoostingClassifier()
gbdt_model.fit(X_scaled, y_train)

# --- DNN Model
model_dnn = Sequential()
model_dnn.add(Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)))
model_dnn.add(Dense(16, activation='relu'))
model_dnn.add(Dense(4, activation='softmax'))  # 4 classes

model_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_scaled, y_train, epochs=20, batch_size=16, verbose=0)

# --- Evaluation
y_pred_gbdt = gbdt_model.predict(X_test_scaled)
y_pred_dnn = np.argmax(model_dnn.predict(X_test_scaled), axis=1)
final_preds = (y_pred_gbdt + y_pred_dnn) // 2  # Simple ensemble logic

accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds, average='weighted')
recall = recall_score(y_test, final_preds, average='weighted')
f1 = f1_score(y_test, final_preds, average='weighted')

# --- Streamlit UI
st.title("Student Performance Predictor")

study_hours = st.slider("Study Hours (per day)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours (per night)", 4, 10, 7)
class_participation = st.slider("Class Participation (1-10)", 1, 10, 5)
assignments_done = st.slider("Assignments Done (1-10)", 1, 10, 5)
quizzes_score = st.slider("Quizzes Score (1-10)", 1, 10, 5)

if st.button("Predict"):
    input_data = np.array([[study_hours, sleep_hours, class_participation, assignments_done, quizzes_score]])
    input_scaled = scaler.transform(input_data)

    pred1 = gbdt_model.predict(input_scaled)
    pred2 = np.argmax(model_dnn.predict(input_scaled), axis=1)

    final = (pred1 + pred2) // 2

    labels = ['Needs Improvement', 'Satisfactory', 'Excellent', 'At Risk']
    st.subheader("Prediction:")
    st.success(labels[final[0]])

    st.subheader("Model Performance Metrics:")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")
