import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_param=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.regularization = regularization
        self.lambda_param = lambda_param

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_regularization_term(self, gradient):
        if self.regularization == 'l1':
            return gradient + (self.lambda_param / len(self.X)) * np.sign(self.weights)
        elif self.regularization == 'l2':
            return gradient + (self.lambda_param / len(self.X)) * self.weights
        else:
            return gradient

    def fit(self, X, y):
        self.X = X
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights)
            predictions = self.sigmoid(linear_model)
            gradient = np.dot(X.T, (predictions - y)) / num_samples
            gradient = self._add_regularization_term(gradient)
            self.weights -= self.learning_rate * gradient
    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        predicted_probabilities = self.sigmoid(linear_model)
        predicted_labels = [1 if p >= 0.5 else 0 for p in predicted_probabilities]
        return predicted_labels

iris = load_iris()
X = iris.data
y = iris.target
y_binary = np.where(y == 0, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print data before training
print("Data before training:")
print("X_train_scaled:", X_train_scaled)
print("y_train:", y_train)

# Train logistic regression model
print("\nTraining logistic regression model...")
model = LogisticRegression(learning_rate=0.01, num_iterations=1000, regularization='l2', lambda_param=0.1)
model.fit(X_train_scaled, y_train)
print("Training complete.")

# Print data after training
print("\nData after training:")
print("Weights:", model.weights)

# Print data before prediction
print("\nData before prediction:")
print("X_test_scaled:", X_test_scaled)

# Make predictions
print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test_scaled)
print("Predictions complete.")

# Print data after prediction
print("\nData after prediction:")
print("y_pred:", y_pred)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
