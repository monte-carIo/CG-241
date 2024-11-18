import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def example_function(x, y):
    """
    A complex function with multiple local minima, maxima, and saddle points.
    """
    # Sinusoidal components for oscillatory behavior
    sinusoidal = torch.sin(2 * x) * torch.cos(2 * y) + torch.sin(y) * torch.cos(x)

    # Gaussian bumps for local extrema
    gaussian_1 = torch.exp(-((x - 2)**2 + (y - 2)**2) / 2)
    gaussian_2 = torch.exp(-((x + 2)**2 + (y + 2)**2) / 1.5)
    gaussian_3 = torch.exp(-((x - 3)**2 + (y + 3)**2) / 1)
    gaussian_4 = torch.exp(-((x + 3)**2 + (y - 3)**2) / 0.2)

    # Polynomial component for global trends and saddle points
    polynomial = (x**3 - x * y**2)

    # Combining all components with weights
    # result =  (gaussian_1 + gaussian_2 - gaussian_3 - gaussian_4) + polynomial
    result = sinusoidal + (gaussian_1 + gaussian_2 - gaussian_3 - gaussian_4)

    return result


# Define a more complex MLP model
class ComplexMLP(nn.Module):
    def __init__(self):
        super(ComplexMLP, self).__init__()

        # Increased the number of neurons in each layer and added an additional hidden layer
        self.layer1 = nn.Linear(2, 16)  # First hidden layer with 16 neurons
        self.layer2 = nn.Linear(16, 32)  # Second hidden layer with 32 neurons
        self.layer3 = nn.Linear(32, 16)  # Third hidden layer with 16 neurons
        self.output_layer = nn.Linear(16, 1)  # Output layer

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.output_layer(x)  # No activation on the output layer (for regression output)
        return x*50
    

def get_point(model, size, num_edge):
    model = example_function
    x_values = np.linspace(-size/2, size/2, num_edge)
    y_values = np.linspace(-size/2, size/2, num_edge)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.zeros_like(X)

    # Calculate the model's output z for each point on the grid
    with torch.no_grad():
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # input_tensor = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32)
                # Z[i, j] = model(input_tensor).item()
                Z[i, j] = model(torch.tensor(X[i, j], dtype=torch.float32),
                                torch.tensor(Y[i, j], dtype=torch.float32)).item()

    # Randomly initialize x and y with requires_grad=True to compute gradients
    # x = torch.tensor([[np.random.randn()]], dtype=torch.float32, requires_grad=True)
    # y = torch.tensor([[np.random.randn()]], dtype=torch.float32, requires_grad=True)
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

def get_trajectory(model, size=15, optimizer='Adam', lr=0.04):
    model = example_function
    x = torch.tensor([[-4]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[-5]], dtype=torch.float32, requires_grad=True)
    # Define the optimizer for x and y
    if optimizer == 'SGD':
        optimizer = optim.SGD([x, y], lr=lr)
    elif optimizer == 'Adam':
        optimizer = optim.Adam([x, y], lr=lr)
    elif optimizer == 'RMSprop':
        optimizer = optim.RMSprop([x, y], lr=lr)
    elif optimizer == 'Adagrad':
        optimizer = optim.Adagrad([x, y], lr=lr)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW([x, y], lr=lr)
    # Store the trajectory for visualization
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    gradient_x = []
    gradient_y = []

    step = 0
    # Optimization loop
    # breakpoint()
    while True:
        # Zero gradients for x and y
        optimizer.zero_grad()

        # Forward pass through the model
        # input_tensor = torch.cat((x, y), dim=1)
        x_pre = x.clone().detach()
        y_pre = y.clone().detach()
        # z = model(input_tensor)  # z is the output, which we'll consider as the loss
        z = model(x, y)

        # Define the custom loss function as simply the output z
        loss = z

        # Backward pass to compute gradients with respect to x and y
        loss.backward()
        
        # Store gradients for x and y
        gradient_x.append(x.grad.item())
        gradient_y.append(y.grad.item())

        # Update x and y using the optimizer
        optimizer.step()

        if (len(trajectory_z) > 1 and(abs(z.item() - trajectory_z[-1]) <1e-6)) or (step > 3000):
            break
        if x.item() < -size/2 or x.item() > size/2 or y.item() < -size/2 or y.item() > size/2:
            break

        # Update the input tensor with new x, y values and store the trajectory
        trajectory_x.append(x_pre.item())
        trajectory_y.append(y_pre.item())
        trajectory_z.append(z.item())
        step += 1
    # (nnumpoints, 3), (num_edge * num_edge, 3)
    return (np.stack((trajectory_x,trajectory_y,trajectory_z), axis=-1),
            np.stack((gradient_x, gradient_y), axis=-1))

if __name__ == "__main__":
    pass