import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset (assuming it's in the same directory as this script)
df = pd.read_csv('Classified Data', index_col=0)

# Feature scaling for KNN
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))  # Fit scaler to features (excluding target)
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

# Create a DataFrame with scaled features
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1]) 

# Split into training and testing sets
X = df_scaled
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Initial KNN model (default k=5)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Evaluate the initial model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Find optimal K value using elbow method
error_rate = []
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plot error rate vs. K value
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()

# Re-train with optimal K
optimal_k = 38 # Based on the elbow plot
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Evaluate the final model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
