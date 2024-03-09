import numpy as np

def sigmoid(x):
 """
 Sigmoid activation function
 """
 return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
 """
 Derivative of the sigmoid activation function
 """
 return sigmoid(x) * (1 - sigmoid(x))

def AND_gate(x1, x2, target):
    """
    AND gate logic using a neural network
    """
    weights = np.array([0.5, 0.5]) # Initialize weights
    bias = -1.5 # Initialize bias
    
    # Forward propagation
    z = np.dot(weights, [x1, x2]) + bias
    h = sigmoid(z)
    
    # Calculate the error
    error = h - target
    
    # Backward propagation
    delta = error * derivative_sigmoid(z) # Corrected calculation of delta
    weight_delta = delta * np.array([x1, x2]) # Corrected calculation of weight_delta
    
    # Update weights and bias based on learning rate
    weights -= learning_rate * weight_delta
    bias -= learning_rate * delta
    
    return h

learning_rate = 0.05
epochs = 1000

# Training data
inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
targets = np.array([0, 0, 0, 1])

for epoch in range(epochs):
 # Forward propagation
 outputs = np.array([AND_gate(x1, x2, target) for x1, x2, target in zip(inputs[:, 0], inputs[:, 1], targets)])
  
 # Error calculation (mean squared error)
 error = np.mean(np.square(targets - outputs))
  
 # Print error for each epoch (optional)
 print(f'Epoch: {epoch+1}, Error: {error}')
  
 # Break loop if error is below convergence threshold
 if error <= 0.002:
    break

print('Training complete!')
