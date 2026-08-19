import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 1. Load dataset
df = pd.read_csv("data/loan_model.csv")
print("Dataset:", df.shape)


# 2. Train/Test split
train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["loan_status"]
)

print("Train:", train.shape)
print("Test:", test.shape)


# 3. Clean DTI
median_dti = train["dti"].median()

train.loc[
    train["dti"].isna() | (train["dti"] == 999),
    "dti"
] = median_dti

test.loc[
    test["dti"].isna() | (test["dti"] == 999),
    "dti"
] = median_dti

print("DTI median:", median_dti)


# 4. Convert employment length
train["emp_length_num"] = (
    train["emp_length"]
    .str.extract(r"(\d+)")[0]
    .astype(float)
)

test["emp_length_num"] = (
    test["emp_length"]
    .str.extract(r"(\d+)")[0]
    .astype(float)
)

train.loc[
    train["emp_length"] == "< 1 year",
    "emp_length_num"
] = 0

test.loc[
    test["emp_length"] == "< 1 year",
    "emp_length_num"
] = 0

train["emp_length_num"] = train["emp_length_num"].fillna(-1)
test["emp_length_num"] = test["emp_length_num"].fillna(-1)


# 5. Convert grade
grade_map = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7
}

train["grade_num"] = train["grade"].map(grade_map)
test["grade_num"] = test["grade"].map(grade_map)


# 6. Create target
risky = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off"
]

train["target"] = train["loan_status"].isin(risky).astype(int)
test["target"] = test["loan_status"].isin(risky).astype(int)

print("Train target:")
print(train["target"].value_counts())

print("Test target:")
print(test["target"].value_counts())


# 7. Select features
features = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "emp_length_num",
    "grade_num"
]

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# 8. Standard Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled train:", X_train_scaled.shape)
print("Scaled test:", X_test_scaled.shape)


# 9. Create Logistic Regression model
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# 10. Train the model
model.fit(X_train_scaled, y_train)

print("Logistic Regression training completed!")


# 11. Make predictions
predictions = model.predict(X_test_scaled)

print("Predictions:", predictions.shape)
print("First 10 predictions:", predictions[:10])

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# Model evaluation
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ROC-AUC uses probabilities, not 0/1 predictions
probabilities = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, probabilities)

print("\nROC-AUC:", roc_auc)

# ROC Curve
RocCurveDisplay.from_predictions(y_test, probabilities)
plt.title("ROC Curve - Logistic Regression V1")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Confusion Matrix - Logistic Regression V1")
plt.show()