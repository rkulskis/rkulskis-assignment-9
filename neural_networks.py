import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

import numpy as np

import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        
        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.activation_fn = activation
        
        # Weights initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # Input to Hidden
        self.b1 = np.zeros((1, hidden_dim))  # Bias for Hidden
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01  # Hidden to Output
        self.b2 = np.zeros((1, output_dim))  # Bias for Output
        
    def _tanh(self, x):
        return np.tanh(x)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Forward pass through the hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear combination
        if self.activation_fn == 'tanh':
            self.a1 = self._tanh(self.z1)  # Activation function
        elif self.activation_fn == 'relu':
            self.a1 = self._relu(self.z1)
        elif self.activation_fn == 'sigmoid':
            self.a1 = self._sigmoid(self.z1)
        
        # Forward pass through the output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._sigmoid(self.z2)  # Sigmoid for binary classification
        
        return self.a2
    
    def backward(self, X, y):
        # Compute loss (binary cross-entropy)
        m = X.shape[0]
        loss = -np.mean(y * np.log(self.a2) + (1 - y) * np.log(1 - self.a2))
        
        # Compute gradients (backpropagation)
        dz2 = self.a2 - y  # Derivative of loss w.r.t. output
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        if self.activation_fn == 'tanh':
            dz1 = np.dot(dz2, self.W2.T) * self._tanh_derivative(self.a1)
        elif self.activation_fn == 'relu':
            dz1 = np.dot(dz2, self.W2.T) * self._relu_derivative(self.a1)
        elif self.activation_fn == 'sigmoid':
            dz1 = np.dot(dz2, self.W2.T) * self._sigmoid_derivative(self.a1)
        
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Save the gradient of the loss with respect to the input (needed for visualization)
        self.dL_dX = dz1  # Gradient of the loss w.r.t. input (from the backward pass)
        
        # Update weights using gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        return loss
    
def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Clear previous plots
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Hidden features
    hidden_features = mlp.a1  # Activations from the hidden layer
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], 
                      hidden_features[:, 2] if hidden_features.shape[1] > 2 else np.zeros_like(hidden_features[:, 0]),
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title('Hidden Layer Features')
    
    # Visualize the decision boundary in the hidden space (hyperplane)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    hidden_space = mlp.forward(grid)
    ax_hidden.contourf(xx, yy, hidden_space.reshape(xx.shape), levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.2)

    # Input space transformed by the hidden layer
    transformed_input = mlp.a1  # Output of the hidden layer
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title('Transformed Input Space')

    # Visualize decision boundary in the input space
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid)
    preds = preds.reshape(xx.shape)
    
    # Use contourf to plot decision boundary in the input space
    ax_input.contourf(xx, yy, preds, levels=np.linspace(0, 1, 10), cmap='coolwarm', alpha=0.2)

    # Visualize features and gradients
    # Compute gradient with respect to input layer
    grad_input = mlp.dL_dX  # Gradient of the loss with respect to the input (from backward pass)
    
    # Represent the gradient as edge thickness by modifying contour levels
    gradient_magnitude = np.linalg.norm(grad_input, axis=1)  # Magnitude of the gradient
    grad_threshold = np.percentile(gradient_magnitude, 90)  # Set threshold for significant gradients
    
    # Plot thicker edges where gradient magnitude is larger
    for i, (x, y) in enumerate(X):
        if gradient_magnitude[i] > grad_threshold:
            ax_gradient.plot([x, X[i, 0] + grad_input[i, 0]], [y, X[i, 1] + grad_input[i, 1]], 
                             'k-', lw=1 + gradient_magnitude[i] * 10, alpha=0.7)

    ax_gradient.set_title('Gradient Magnitude and Features')
    ax_gradient.set_xlim(-1.5, 1.5)
    ax_gradient.set_ylim(-1.5, 1.5)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
