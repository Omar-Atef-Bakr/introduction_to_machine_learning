import numpy as np

# Data points and labels (2-dimensional dataset with 8 points and 2 classes)
X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5],
              [1, 0],
              [2, 1],
              [4, 3],
              [5, 4]])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Function to calculate the sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to calculate the cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initial weights and biases
w0 = -2
w1 = -2
w2 = 1

# Learning rate
learning_rate = 0.1

# Number of epochs
num_epochs = 3

# Training loop for 3 epochs
for epoch in range(num_epochs):
    # Forward propagation
    z = w0 + w1 * X[:, 0] + w2 * X[:, 1]
    y_pred = sigmoid(z)

    # Compute the cross-entropy loss
    loss = cross_entropy_loss(y, y_pred)

    # Compute the gradients with respect to the weights and biases
    d_loss_d_w0 = np.mean(y_pred - y)
    d_loss_d_w1 = np.mean((y_pred - y) * X[:, 0])
    d_loss_d_w2 = np.mean((y_pred - y) * X[:, 1])

    # Update the weights and biases using gradient descent
    w0 -= learning_rate * d_loss_d_w0
    w1 -= learning_rate * d_loss_d_w1
    w2 -= learning_rate * d_loss_d_w2

    # Compute and report the training accuracy after each epoch
    predictions = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Training Accuracy = {accuracy:.2f}")

# Final weights after training
print("\nFinal Weights:")
print(f"w0 = {w0:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f}")