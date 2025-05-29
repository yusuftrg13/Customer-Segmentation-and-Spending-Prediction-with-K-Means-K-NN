import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Encode categorical variables
le = LabelEncoder()
df['Genre_Encoded'] = le.fit_transform(df['Genre'])

# Feature selection for classification
# We'll try to predict spending score categories based on other features

# Create spending score categories (Low: 0-33, Medium: 34-66, High: 67-100)
df['Spending_Category'] = pd.cut(df['Spending Score (1-100)'], 
                                bins=[0, 33, 66, 100], 
                                labels=['Low', 'Medium', 'High'])

# Encode the target variable
df['Spending_Category_Encoded'] = le.fit_transform(df['Spending_Category'])

# Select features and target
X = df[['Age', 'Annual Income (k$)', 'Genre_Encoded']].values
y = df['Spending_Category_Encoded'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Find the optimal K value using cross-validation
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot the cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy for different values of K')
plt.grid(True)
plt.savefig('knn_cv_accuracy.png')

# Choose the best K value (the one with highest cross-validation accuracy)
best_k = k_range[cv_scores.index(max(cv_scores))]
print(f"\nBest K value from cross-validation: {best_k}")

# Fine-tune the model with GridSearchCV
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"\nBest parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train the K-NN model with the best parameters
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)

# Make predictions
y_pred = best_knn.predict(X_test)

# Evaluate the model
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Feature importance analysis
# For K-NN, we can analyze feature importance by looking at the impact of each feature on prediction

# Train models with individual features and compare accuracy
feature_names = ['Age', 'Annual Income', 'Gender']
feature_scores = []

for i in range(X.shape[1]):
    # Create a dataset with only one feature
    X_single = X_scaled[:, i].reshape(-1, 1)
    
    # Split the data
    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single, y, test_size=0.3, random_state=42)
    
    # Train the model
    knn_single = KNeighborsClassifier(n_neighbors=best_k)
    knn_single.fit(X_train_single, y_train_single)
    
    # Evaluate the model
    y_pred_single = knn_single.predict(X_test_single)
    accuracy = accuracy_score(y_test_single, y_pred_single)
    feature_scores.append(accuracy)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_names, y=feature_scores)
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.title('Feature Importance based on Individual Accuracy')
plt.savefig('feature_importance.png')

# Visualize the relationship between features and spending categories
plt.figure(figsize=(15, 10))

# Age vs Spending Category
plt.subplot(2, 2, 1)
sns.boxplot(x='Spending_Category', y='Age', data=df)
plt.title('Age vs Spending Category')

# Annual Income vs Spending Category
plt.subplot(2, 2, 2)
sns.boxplot(x='Spending_Category', y='Annual Income (k$)', data=df)
plt.title('Annual Income vs Spending Category')

# Gender vs Spending Category
plt.subplot(2, 2, 3)
sns.countplot(x='Spending_Category', hue='Genre', data=df)
plt.title('Gender vs Spending Category')

# Age and Annual Income vs Spending Category
plt.subplot(2, 2, 4)
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='Spending_Category', data=df)
plt.title('Age and Annual Income vs Spending Category')

plt.tight_layout()
plt.savefig('feature_relationships.png')

# Predict spending category for new customers
def predict_spending_category(age, income, gender):
    # Encode gender
    gender_encoded = 1 if gender.lower() == 'female' else 0
    
    # Create feature array
    features = np.array([[age, income, gender_encoded]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = best_knn.predict(features_scaled)
    
    # Decode prediction
    spending_category = ['Low', 'Medium', 'High'][prediction[0]]
    
    return spending_category

# Example predictions
print("\nExample Predictions:")
print(f"Customer 1 (Age: 25, Income: 50k, Gender: Female) - Predicted Spending Category: {predict_spending_category(25, 50, 'Female')}")
print(f"Customer 2 (Age: 45, Income: 70k, Gender: Male) - Predicted Spending Category: {predict_spending_category(45, 70, 'Male')}")
print(f"Customer 3 (Age: 35, Income: 30k, Gender: Female) - Predicted Spending Category: {predict_spending_category(35, 30, 'Female')}")

# Print conclusions
print("\nConclusions:")
print("1. The K-NN algorithm successfully predicts customer spending categories based on age, income, and gender.")
print(f"2. The model achieved an accuracy of {accuracy_score(y_test, y_pred):.2f} on the test set.")
print(f"3. The most important feature for predicting spending category is {feature_names[feature_scores.index(max(feature_scores))]}.")
print("4. This model can be used to identify potential high-spending customers for targeted marketing campaigns.")