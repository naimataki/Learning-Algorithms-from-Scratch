import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def compute_loss(theta, X_b, y): #expect X_b (X with bias)
    m = X_b.shape[0]
    predictions = X_b.dot(theta)
    errors = predictions - y
    loss = (1/(2*m)) * (errors.T.dot(errors))
    return loss

def gradient_step(X_b, y, theta, alpha):
    m = X_b.shape[0]
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X_b.T.dot(errors)
    theta_updated = theta - alpha * gradient
    return theta_updated

def train_model(X, y, alpha=0.01, epochs=1000, method='batch'):
    loss_history = []

    X_norm, mean, std = normalize(X) #normalize
    X_norm_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm] #Shape (m, n+1)

    n = X_norm_b.shape[1] #number of features + 1 for bias
    theta = np.zeros(n)

    m = X_norm_b.shape[0]

    if method == 'batch':
        for epoch in range(epochs):
            loss = compute_loss(theta, X_norm_b, y)
            loss_history.append(loss)
            theta = gradient_step(X_norm_b, y, theta, alpha)
    
    elif method == 'sgd':
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_norm_b_shuffled = X_norm_b[indices]
            y_shuffled = y[indices]
            epoch_loss_sum = 0
            for i in range(m):
                xi = X_norm_b_shuffled[i:i+1] #shape (1, n+1)
                yi = y_shuffled[i:i+1] #shape (1,)
                theta = gradient_step(xi, yi.flatten(), theta, alpha)
            loss = compute_loss(theta, X_norm_b_shuffled, y_shuffled)
            loss_history.append(loss)

    return theta, loss_history, mean, std

def generate_synthetic_data(n_samples = 100, theta_true = 2, noise_std=0.1, bias=4):
    # Step 1: Generate random feature matrix X (n_samples x 1 feature)
    X = np.random.rand(n_samples, 1) * 10 # (Random values between 0 and 1) x 10 for better viz

    # Step 2: Compute the true target values y without noise (y = X * theta)
    theta_full_true = np.array([[bias], [theta_true]]) #[[4], [2]]
    X_b = np.c_[np.ones((n_samples, 1)), X]
    y_true = X_b.dot(theta_full_true)  # Linear relationship (matrix multiplication)
    # Step 3: Add noise to the target values (random Gaussian noise)
    noise = np.random.randn(n_samples, 1) * noise_std
    y = (y_true + noise).flatten()  # Final target values with noise, flatten to !D array (100,)
    return X, y #return original X

def plot_regression_line(X, y, theta, mean, std):
    
    # Prepare X for prediction (apply the same steps as in training)
    # 1. Normalize X using the *training* mean and std
    X_norm = (X - mean) / (std + 1e-8) # Add epsilon for safety
    
    # 2. Add bias term
    X_norm_b = X_norm.c_[np.ones((X_norm.shape[0], 1)), X_norm]

    # 3. Make predictions using learned theta
    y_pred = X_norm_b.dot(theta)
    
    # Plot original data points
    plt.scatter(X.flatten(), y.flatten(), color='blue', label='Data Points')
    
    # Plot the regression line
    plt.plot(X.flatten(), y.flatten(), color='red', label='Regression Line')
    
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