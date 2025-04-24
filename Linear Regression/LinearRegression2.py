import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def compute_loss(theta, X, y):
    m = X.shape[0]
    predictions = X.dot(theta)
    errors = predictions - y
    loss = (1/(2*m)) * (errors.T.dot(errors))
    return loss

def gradient_step(X, y, theta, alpha):
    m = X.shape[0]
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X.T.dot(errors)
    theta_updated = theta - alpha * gradient
    return theta_updated

def train_model(X, y, alpha=0.01, epochs=1000, method='batch'):
    loss_history = []
    n = X.shape[1]
    theta = np.zeros(n)

    X_norm, _, _ = normalize(X)

    if method == 'batch':
        for epoch in range(epochs):
            loss = compute_loss(theta, X_norm, y)
            loss_history.append(loss)
            theta = gradient_step(X_norm, y, theta, alpha)
    
    elif method == 'sgd':
        for epoch in range(epochs):
            indices = np.random.permutation(X_norm.shape[0])
            X_norm_shuffled = X_norm[indices]
            y_shuffled = y[indices]
            for i in range(X_norm_shuffled.shape[0]):
                xi = X_norm_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                theta = gradient_step(xi, yi, theta, alpha)
            loss = compute_loss(theta, X_norm_shuffled, y_shuffled)
            loss_history.append(loss)

    return theta, loss_history

def generate_synthetic_data(n_samples = 100, theta_true = 2, noise_std=0.1, bias=4):
    # Step 1: Generate random feature matrix X (n_samples x 1 feature)
    X = np.random.rand(n_samples, 1)  # Random values between 0 and 1

    # Step 2: Compute the true target values y without noise (y = X * theta)
    y_true = X.dot(np.array([theta_true]))  # Linear relationship

    # Step 3: Add noise to the target values (random Gaussian noise)
    noise = np.random.randn(n_samples, 1) * noise_std
    y = (y_true + noise + bias).flatten()  # Final target values with noise
    return X, y 

def plot_regression_line(X, y, theta):
    # Flatten X and y for plotting
    X_flat = X.flatten()
    y_flat = y.flatten()
    
    # Make predictions - need to reshape X_flat for dot product
    X_for_pred = X_flat.reshape(-1, 1)  # Reshape to column vector
    y_pred = X_for_pred.dot(theta)
    y_pred_flat = y_pred.flatten()  # Flatten predictions for plotting
    
    # Plot original data points
    plt.scatter(X_flat, y_flat, color='blue', label='Data Points')
    
    # Plot the regression line
    plt.plot(X_flat, y_pred_flat, color='red', label='Regression Line')
    
    # Labels and title
    plt.xlabel('Feature (X)')
    plt.ylabel('Target (y)')
    plt.title('Linear Regression Line')
    
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()

def plot_loss_curve(loss_history):
    # Create a range of epochs (or iterations)
    epochs = range(len(loss_history))
    
    # Plot the loss history
    plt.plot(epochs, loss_history, color='blue', label='Loss')
    
    # Labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve over Epochs')
    
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()

X, y = generate_synthetic_data(n_samples = 100, theta_true = 2, noise_std=0.1, bias=4)
theta, loss_history = train_model(X, y, alpha=0.01, epochs=500, method='batch')
print(loss_history[-1])
plot_regression_line(X, y, theta)
plot_loss_curve(loss_history)