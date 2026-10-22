
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate sample student data
rng = np.random.default_rng(42)
n = 1000

students = pd.DataFrame({
    "study_hours": rng.uniform(0, 10, n).round(1),
    "attendance": rng.uniform(40, 100, n).round(1),
    "previous_score": rng.uniform(20, 100, n).round(1),
    "assignments_completed": rng.integers(0, 11, n),
    "sleep_hours": rng.uniform(3, 10, n).round(1),
})

# 2. Create a synthetic outcome rule with some randomness
score = (
    students["study_hours"] * 4
    + students["attendance"] * 0.20
    + students["previous_score"] * 0.35
    + students["assignments_completed"] * 1.5
    + students["sleep_hours"] * 0.8
    + rng.normal(0, 8, n)
)

students["result"] = np.where(score >= 45, "Pass", "Fail")

# Save sample data
students.to_csv("students.csv", index=False)

# 3. Select features and target
X = students.drop(columns=["result"])
y = students["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)

print("Model training completed!")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# 6. Save model and feature names
joblib.dump(
    {
        "model": model,
        "features": list(X.columns),
        "classes": list(model.classes_),
    },
    "student_model.joblib"
)

print("\nSaved: students.csv")
print("Saved: student_model.joblib")
