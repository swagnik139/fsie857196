import os
import json
import pandas as pd
import joblib
import tarfile
from sklearn.metrics import accuracy_score

# Define paths
model_dir = "/opt/ml/processing/model"
model_tar_path = os.path.join(model_dir, "model.tar.gz")
test_path = "/opt/ml/processing/test/test.csv"
output_path = "/opt/ml/processing/evaluation"

# Extract model.tar.gz
with tarfile.open(model_tar_path) as tar:
    tar.extractall(path=model_dir)

# Load the extracted model
model = joblib.load(os.path.join(model_dir, "model.joblib"))

# Load the test data
test_data = pd.read_csv(test_path)
X_test = test_data.drop("price_range", axis=1)
y_test = test_data["price_range"]

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Save the evaluation report
report_dict = {
    "binary_classification_metrics": {
        "accuracy": {
            "value": accuracy
        }
    }
}

# Write evaluation to JSON
os.makedirs(output_path, exist_ok=True)
with open(os.path.join(output_path, "evaluation.json"), "w") as f:
    json.dump(report_dict, f)

print(f"Accuracy: {accuracy:.4f}")
