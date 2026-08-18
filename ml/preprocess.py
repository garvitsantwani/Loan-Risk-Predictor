import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   
df = pd.read_csv("../data/loan_model.csv")

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["loan_status"]
)

print("Train:", train.shape)
print("Test:", test.shape)


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
print("Train DTI missing:", train["dti"].isna().sum())
print("Test DTI missing:", test["dti"].isna().sum())
train["emp_length_num"] = train["emp_length"].str.extract(r"(\d+)")[0].astype(float)
test["emp_length_num"] = test["emp_length"].str.extract(r"(\d+)")[0].astype(float)

train.loc[train["emp_length"] == "< 1 year", "emp_length_num"] = 0
test.loc[test["emp_length"] == "< 1 year", "emp_length_num"] = 0

train["emp_length_num"] = train["emp_length_num"].fillna(-1)
test["emp_length_num"] = test["emp_length_num"].fillna(-1)

print("Train emp_length missing:", train["emp_length_num"].isna().sum())
print("Test emp_length missing:", test["emp_length_num"].isna().sum())

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

print("Train grade missing:", train["grade_num"].isna().sum())
print("Test grade missing:", test["grade_num"].isna().sum())

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

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled train shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)