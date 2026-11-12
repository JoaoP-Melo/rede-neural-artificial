import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

np.random.seed(42)

iris = load_iris()
X = iris.data                      
y = iris.target                   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

def one_hot(labels, n_classes=3):
    encoded = np.zeros((labels.shape[0], n_classes))
    encoded[np.arange(labels.shape[0]), labels] = 1
    return encoded

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

n_input, n_hidden, n_output = 4, 8, 3


W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
b2 = np.zeros((1, n_output))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(Y_true, Y_pred):
    eps = 1e-12  # evita log(0)
    return -np.mean(np.sum(Y_true * np.log(Y_pred + eps), axis=1))

def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def accuracy(Y_true, Y_pred):
    return np.mean(np.argmax(Y_true, axis=1) == np.argmax(Y_pred, axis=1))

epochs = 1000
learning_rate = 0.1
m = X_train.shape[0]

for epoch in range(1, epochs + 1):
    Z1, A1, Z2, A2 = forward(X_train)

    loss = cross_entropy_loss(Y_train, A2)

    dZ2 = (A2 - Y_train) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 100 == 0 or epoch == 1:
        train_acc = accuracy(Y_train, A2)
        print(f"Época {epoch:4d}/{epochs} | Loss: {loss:.4f} | Acurácia treino: {train_acc * 100:.2f}%")

_, _, _, A2_test = forward(X_test)
test_loss = cross_entropy_loss(Y_test, A2_test)
test_acc = accuracy(Y_test, A2_test)

print("\n===== Avaliação Final =====")
print(f"Loss no teste: {test_loss:.4f}")
print(f"Acurácia no teste: {test_acc * 100:.2f}%")