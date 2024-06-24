import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset 
df = pd.read_csv('KNN_Project_Data')

# Feature Scaling
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))  # Fit to all features except the target
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

# Create a DataFrame with scaled features
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Split into training and testing sets
X = df_feat
y = df['TARGET CLASS'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Initial KNN model 
knn = KNeighborsClassifier()  
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Evaluate initial model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Elbow Method to find optimal K
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Plot error rates
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()


# Retrain with optimal k value
knn = KNeighborsClassifier(n_neighbors=31) 
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Evaluate final model
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))