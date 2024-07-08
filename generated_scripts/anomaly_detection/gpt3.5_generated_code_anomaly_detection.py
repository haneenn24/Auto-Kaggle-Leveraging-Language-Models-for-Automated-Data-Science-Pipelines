# Load dataset (assuming creditcard.csv is in the current directory)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Initial data exploration
print("Dataset structure and summary statistics:")
print(df.head())
print(df.describe())

# Splitting data into features and target variable
X = df.drop('Class', axis=1)  # Features (excluding 'Class' column)
y = df['Class']  # Target variable 'Class' (0: normal, 1: fraudulent)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training (example with Isolation Forest)
model = IsolationForest(contamination='auto', random_state=42)
model.fit(X_train)

# Predicting anomaly scores on the test set
y_pred_scores = model.decision_function(X_test)

# Evaluating the model using ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_scores)
print(f"\nROC AUC score: {roc_auc}")
