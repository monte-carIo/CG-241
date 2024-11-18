import random
from copy import deepcopy
from sympy import *

def generate_points(size, num_edge):
    """Generate points for mesh grid"""
    step = size / num_edge
    return [[-size / 2 + j * step, size / 2 - i * step, 0] for i in range(num_edge + 1) for j in range(num_edge + 1)]

def evaluate_points(points, expression_str):
    """Evaluate Z-values for points based on a user-defined mathematical expression"""
    x, y = symbols("x y")
    expression = eval(expression_str)
    mean_z = 0

    evaluated_points = []
    for point in points:
        point[2] = expression.subs([(x, point[0]), (y, point[1])])
        mean_z += point[2]
        evaluated_points.append(point)
    
    mean_z /= len(points)
    return evaluated_points, mean_z

def compute_gradient(point, gradient_func):
    """Compute gradient for a given point using the gradient function"""
    grad_x = gradient_func[0](*point[:2])  # Evaluate the gradient function for x
    grad_y = gradient_func[1](*point[:2])  # Evaluate the gradient function for y
    return grad_x, grad_y

def gradient_descent(rate, expression_str):
    """Compute points and gradients for gradient descent"""
    points = []
    gradients = []

    # Create symbolic variables and parse the user-defined expression
    x, y = symbols("x y")
    expression = eval(expression_str)

    # Compute symbolic gradients with respect to x and y
    grad_x_expr = expression.diff(x)
    grad_y_expr = expression.diff(y)

    # Convert symbolic gradients to numerical functions
    gradient_func = (lambdify((x, y), grad_x_expr, "numpy"), 
                     lambdify((x, y), grad_y_expr, "numpy"))

    # Random starting point and initial evaluation
    x_val, y_val = random.uniform(-15/2, 15/2), random.uniform(-15/2, 15/2)
    initial_point, _ = evaluate_points([[x_val, y_val, 0]], expression_str)
    points.append(deepcopy(initial_point[0]))

    # Gradient descent loop
    while len(points) < 6 * round(abs(x_val) / rate + 8 * abs(y_val) / rate):
        # Compute gradient based on current point
        grad_x, grad_y = compute_gradient(initial_point[0], gradient_func)
        gradients.append([grad_x, grad_y])

        # Update point based on gradient
        initial_point[0][0] -= rate * grad_x
        initial_point[0][1] -= rate * grad_y
        initial_point, _ = evaluate_points(initial_point, expression_str)
        points.append(deepcopy(initial_point[0]))
        if (abs(points[-1][2] - points[-2][2]) < 1e-4):
            break

    gradients.append([0, 0])
    return points, gradients