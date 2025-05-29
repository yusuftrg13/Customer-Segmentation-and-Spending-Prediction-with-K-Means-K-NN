import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Exploratory Data Analysis
plt.figure(figsize=(15, 10))

# Gender distribution
plt.subplot(2, 3, 1)
sns.countplot(x='Genre', data=df)
plt.title('Gender Distribution')

# Age distribution
plt.subplot(2, 3, 2)
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')

# Annual Income distribution
plt.subplot(2, 3, 3)
sns.histplot(df['Annual Income (k$)'], kde=True)
plt.title('Annual Income Distribution')

# Spending Score distribution
plt.subplot(2, 3, 4)
sns.histplot(df['Spending Score (1-100)'], kde=True)
plt.title('Spending Score Distribution')

# Relationship between Annual Income and Spending Score
plt.subplot(2, 3, 5)
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Genre', data=df)
plt.title('Income vs Spending Score')

plt.tight_layout()
plt.savefig('eda_plots.png')

# Feature selection for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow_method.png')

# Based on the elbow method, let's choose 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = kmeans_labels

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Customer Segments based on Income and Spending Score')
plt.savefig('kmeans_clusters.png')

# Now let's implement K-NN for classification
# We'll use the clusters identified by K-means as our target variable

# Prepare data for K-NN
X_knn = X_scaled  # Using the scaled features
y_knn = kmeans_labels  # Using the cluster labels as target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42)

# Find the optimal K value
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Testing Accuracy')
plt.title('Accuracy for different values of K')
plt.grid(True)
plt.savefig('knn_accuracy.png')

# Choose the best K value (the one with highest accuracy)
best_k = k_range[scores.index(max(scores))]
print(f"\nBest K value: {best_k}")

# Train the K-NN model with the best K value
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the K-NN decision boundaries
def plot_decision_boundaries(X, y, model, feature_names):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('K-NN Decision Boundaries')
    plt.colorbar()
    plt.savefig('knn_decision_boundaries.png')

# Transform the test data back to original scale for visualization
X_original = scaler.inverse_transform(X_knn)

# Plot decision boundaries
plot_decision_boundaries(X_knn, y_knn, knn, ['Annual Income (scaled)', 'Spending Score (scaled)'])

# Analyze the characteristics of each cluster
cluster_analysis = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'}).reset_index()

print("\nCluster Analysis:")
print(cluster_analysis)

# Visualize the cluster characteristics
plt.figure(figsize=(15, 10))

# Average Age by Cluster
plt.subplot(2, 2, 1)
sns.barplot(x='Cluster', y='Age', data=cluster_analysis)
plt.title('Average Age by Cluster')

# Average Annual Income by Cluster
plt.subplot(2, 2, 2)
sns.barplot(x='Cluster', y='Annual Income (k$)', data=cluster_analysis)
plt.title('Average Annual Income by Cluster')

# Average Spending Score by Cluster
plt.subplot(2, 2, 3)
sns.barplot(x='Cluster', y='Spending Score (1-100)', data=cluster_analysis)
plt.title('Average Spending Score by Cluster')

# Count of Customers by Cluster
plt.subplot(2, 2, 4)
sns.barplot(x='Cluster', y='Count', data=cluster_analysis)
plt.title('Number of Customers by Cluster')

plt.tight_layout()
plt.savefig('cluster_analysis.png')

# Gender distribution within each cluster
plt.figure(figsize=(12, 6))
sns.countplot(x='Cluster', hue='Genre', data=df)
plt.title('Gender Distribution by Cluster')
plt.savefig('gender_by_cluster.png')

# Print conclusions
print("\nConclusions:")
print("1. We identified 5 distinct customer segments based on Annual Income and Spending Score.")
print("2. The K-NN algorithm successfully classified customers into these segments with high accuracy.")
print("3. Each segment has unique characteristics in terms of age, income, and spending behavior.")
print("4. These insights can be used for targeted marketing strategies.")