import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tpot import TPOTClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = './data/customer_churn.csv'
data = pd.read_csv(file_path)

# Drop the columns that are not useful for prediction
data = data.drop(['Names', 'Onboard_date', 'Location', 'Company'], axis=1)

# Encode categorical features (if any)
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the dataset into features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the AutoML classifier using TPOT
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42, n_jobs=-1)

# Train the AutoML model
tpot.fit(X_train, y_train)

# Predict the test set results
y_pred = tpot.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the accuracy and classification report
return accuracy, report

#performance of generated code:
# Accuracy: 87.22%
#               precision    recall  f1-score   support

#            0       0.89      0.97      0.93       148
#            1       0.74      0.44      0.55        32

#     accuracy                           0.87       180
#    macro avg       0.81      0.70      0.74       180
# weighted avg       0.86      0.87      0.86       180

# The model performs well in predicting non-churn cases with high precision and recall. 
# However, it struggles more with predicting churn cases, as indicated by the lower precision,
# recall, and F1-score for class 1.