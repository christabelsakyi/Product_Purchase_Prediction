import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('data/customer_purchase_behavior_200.csv')

# Display first few rows
print(df.head())

# Handle missing values
df.dropna(inplace=True)

# Convert categorical variable (Gender: Male = 0, Female = 1)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Normalize Age and Income using Min-Max Scaling
scaler = MinMaxScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

# Save preprocessed dataset
df.to_csv('data/customer_purchase_behavior_2000.csv', index=False)
print("Preprocessed dataset saved as 'data/customer_purchase_behavior_2000.csv'.")

# Splitting dataset into features and target variable
X = df[['Age', 'Income', 'Gender']]
y = df['Buy_Product']

# Splitting into train and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
}

dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearch
best_params = grid_search.best_params_
print(f"Best Decision Tree Parameters: {best_params}")

# Train Decision Tree with best parameters
best_dt = DecisionTreeClassifier(**best_params, random_state=42, class_weight='balanced')
best_dt.fit(X_train, y_train)

# Predictions
y_pred_dt = best_dt.predict(X_test)

# Evaluate Model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(best_dt, feature_names=['Age', 'Income', 'Gender'], class_names=['No Purchase', 'Purchase'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# 5-Fold Cross-Validation
cv_scores = cross_val_score(best_dt, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

# Making Predictions for New Customers
new_customers = pd.DataFrame([
    [40, 50, 0],  # Male customer
    [30, 45, 1]   # Female customer
], columns=['Age', 'Income', 'Gender'])

# Apply Min-Max Scaling to new customers' Age and Income
new_customers[['Age', 'Income']] = scaler.transform(new_customers[['Age', 'Income']])

# Predict Purchase Decision
new_customer_predictions = best_dt.predict(new_customers)

print(f"Decision Tree - Predicted Purchase Decision for New Customers: {new_customer_predictions}")
