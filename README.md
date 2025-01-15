# Holographic-display
# Install necessary libraries
!pip install scikit-learn matplotlib seaborn pandas numpy zipfile

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from google.colab import files

# Enable MATLAB-style plotting
plt.style.use('classic')

# Step 1: Upload the file manually
uploaded = files.upload()

# Extract the name of the uploaded zip file (assuming it's 'IOT.zip')
zip_file_path = 'IOT.zip'  # Update if the file name is different

# Step 2: Extract the ZIP file
extract_folder = 'extracted_data'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Step 3: Load IoT dataset from extracted files
train_data_path = os.path.join(extract_folder, 'iot_device_train.csv')  # Adjust if needed
test_data_path = os.path.join(extract_folder, 'iot_device_test.csv')    # Adjust if needed

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Check the first few rows of the train and test data
print("Training Data:\n", train_data.head())
print("Test Data:\n", test_data.head())

# Step 4: Data Preprocessing
# Select relevant features (adjust as per your dataset)
selected_features = ['bytes', 'bytes_A', 'bytes_A_B_ratio', 'bytes_B', 'http_GET', 'http_POST', 'http_bytes_avg',
                     'http_bytes_entropy', 'http_bytes_max', 'http_bytes_median', 'http_bytes_min', 'http_bytes_stdev',
                     'http_bytes_sum', 'http_cookie_count', 'http_cookie_values_avg', 'http_cookie_values_entropy',
                     'http_cookie_values_max', 'http_cookie_values_median', 'http_cookie_values_min',
                     'http_cookie_values_stdev', 'http_cookie_values_sum', 'http_cookie_values_thirdQ',
                     'http_cookie_values_var', 'http_count_host', 'http_count_req_content_type',
                     'http_count_resp_code', 'http_count_resp_content_type', 'http_count_transactions',
                     'http_count_user_agents', 'http_dom_host_alexaRank', 'http_dom_resp_code',
                     'http_has_location', 'http_has_referrer', 'http_has_req_content_type',
                     'http_has_resp_content_type', 'http_has_user_agent', 'http_inter_arrivel_avg',
                     'http_inter_arrivel_entropy', 'http_inter_arrivel_firstQ', 'http_inter_arrivel_max',
                     'http_inter_arrivel_median', 'http_inter_arrivel_min', 'http_inter_arrivel_stdev',
                     'http_inter_arrivel_sum', 'http_inter_arrivel_thirdQ', 'http_inter_arrivel_var',
                     'http_req_bytes_avg', 'http_req_bytes_entropy', 'http_req_bytes_firstQ', 'http_req_bytes_max',
                     'http_req_bytes_median', 'http_req_bytes_min', 'http_req_bytes_stdev', 'http_req_bytes_sum',
                     'http_req_bytes_thirdQ', 'http_req_bytes_var', 'http_resp_bytes_avg', 'http_resp_bytes_entropy',
                     'http_resp_bytes_firstQ', 'http_resp_bytes_max', 'http_resp_bytes_median', 'http_resp_bytes_min',
                     'http_resp_bytes_stdev', 'http_resp_bytes_sum', 'http_resp_bytes_thirdQ', 'http_resp_bytes_var',
                     'http_time_avg', 'http_time_entropy', 'http_time_firstQ', 'http_time_max', 'http_time_median',
                     'http_time_min', 'http_time_stdev', 'http_time_sum', 'http_time_thirdQ', 'http_time_var',
                     'is_ssl', 'is_http', 'is_g_http', 'is_cdn_http', 'is_img_http', 'is_ad_http', 'is_numeric_url_http',
                     'is_numeric_url_with_port_http', 'is_tv_http', 'is_cloud_http', 'device_category']

# Extract features and labels from the training data
train_data = train_data[selected_features].dropna()

# Normalize Features
scaler = StandardScaler()
X = train_data.drop('device_category', axis=1)  # Exclude label column
X_scaled = scaler.fit_transform(X)

# Labels
y = train_data['device_category']

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Quantum-Safe Encryption (Simulation)
q = 97  # Small prime modulus
A = np.random.randint(1, q, size=(86, 86))  # Adjusting matrix A to match the number of features (86)
K = np.linalg.inv(A) % q  # Simplified decryption key

def encrypt_data(data, A, q):
    return np.dot(data, A) % q

def decrypt_data(data, K, q):
    return np.round(np.dot(data, K) % q).astype(int)

# Example encryption simulation
encrypted_sample = encrypt_data(X_train[:5], A, q)  # Now it should work with 86 features
decrypted_sample = decrypt_data(encrypted_sample, np.linalg.inv(A), q)

print("Encrypted Data Sample:\n", encrypted_sample)
print("Decrypted Data Sample:\n", decrypted_sample)

# Step 7: Anomaly Detection using Isolation Forest
isolation_model = IsolationForest(contamination=0.05, random_state=42)
isolation_model.fit(X_train)
y_pred = isolation_model.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary labels

# Convert labels to numeric (if they are strings)
y_test = pd.to_numeric(y_test, errors='coerce')  # Ensure y_test is numeric
y_pred = pd.to_numeric(y_pred, errors='coerce')  # Ensure y_pred is numeric

# Reset indices
y_test = y_test.reset_index(drop=True)
y_pred = pd.Series(y_pred).reset_index(drop=True)

# Now you can safely use them
valid_indices = ~pd.isna(y_test) & ~pd.isna(y_pred)
y_test_clean = y_test[valid_indices]
y_pred_clean = y_pred[valid_indices]

# Evaluate
print("Isolation Forest Evaluation:")
print(classification_report(y_test_clean, y_pred_clean))

# Step 9: Comparison with Existing Model (Decision Tree)
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_dt_pred = decision_tree.predict(X_test)

print("Decision Tree Evaluation:")
print(classification_report(y_test, y_dt_pred))

# Step 10: F1-Score Comparison
f1_isolation = f1_score(y_test, y_pred)
f1_decision_tree = f1_score(y_test, y_dt_pred)

print(f"F1-Score (Isolation Forest): {f1_isolation}")
print(f"F1-Score (Decision Tree): {f1_decision_tree}")

# Step 11: Visualization of Results
plt.figure(figsize=(10, 6))
plt.bar(['Isolation Forest', 'Decision Tree'], [f1_isolation, f1_decision_tree], color=['blue', 'orange'])
plt.title('Model Comparison: F1-Score')
plt.ylabel('F1-Score')
plt.grid(True)
plt.show()

# Step 12: Anomaly Detection Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=train_data.index, y=train_data['bytes'], hue=y_pred, palette='coolwarm')
plt.title('Anomaly Detection Results')
plt.xlabel('Data Index')
plt.ylabel('Bytes')
plt.legend(title='Anomaly (1: Yes, 0: No)')
plt.grid(True)
plt.show()

# Step 13: Encryption Overhead Simulation
def encryption_overhead_simulation(data, A, q):
    times = []
    for _ in range(10):
        start = time.time()
        _ = encrypt_data(data, A, q)
        end = time.time()
        times.append(end - start)
    return np.mean(times)

overhead = encryption_overhead_simulation(X_train[:10], A, q)
print("Average Encryption Time Overhead (seconds):", overhead)

plt.figure(figsize=(10, 6))
plt.plot([i for i in range(10)], [encryption_overhead_simulation(X_train[i:i+10], A, q) for i in range(10)], marker='o')
plt.title('Encryption Overhead Across Data Batches')
plt.xlabel('Batch Index')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.show()
