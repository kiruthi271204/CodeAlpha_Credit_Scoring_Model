import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
data = pd.read_csv("credit_data.csv")

# Features and target
X = data[["income", "age", "loan_amount", "credit_history", "employment_years"]]
y = data["approved"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

# Display results
print("===== CREDIT SCORING MODEL =====")
print("Model: Logistic Regression")
print("--------------------------------")
print("Accuracy  :", round(accuracy, 2))
print("Precision :", round(precision, 2))
print("Recall    :", round(recall, 2))
print("F1-Score  :", round(f1, 2))
print("ROC-AUC   :", round(roc_auc, 2))

# Example prediction
new_customer = [[40000, 30, 140000, 1, 5]]
prediction = model.predict(new_customer)

print("--------------------------------")
if prediction[0] == 1:
    print("New Customer: Loan Approved")
else:
    print("New Customer: Loan Not Approved")
