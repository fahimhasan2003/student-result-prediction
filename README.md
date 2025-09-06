# student-result-prediction
# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Create dataset
data = {
    'hours_studied': [2, 4, 6, 8, 10, 1, 3, 5, 7, 9],
    'attendance': [60, 70, 80, 90, 95, 50, 65, 75, 85, 92],
    'result': [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print("📌 Dataset:")
print(df)

# Step 3: Data visualization
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="hours_studied", y="attendance", hue="result", s=100, palette="Set2")
plt.title("Student Result Visualization (Pass/Fail)")
plt.xlabel("Hours Studied")
plt.ylabel("Attendance (%)")
plt.show()

# Step 4: Features & Target
X = df[['hours_studied', 'attendance']]
y = df['result']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: New Student Prediction
new_student = [[6, 85]]  # Example: studied 6 hours, attendance 85%
prediction = model.predict(new_student)

print("\n🎯 Prediction for new student (6 hrs study, 85% attendance):",
      "PASS 🎉" if prediction[0] == 1 else "FAIL ❌")
