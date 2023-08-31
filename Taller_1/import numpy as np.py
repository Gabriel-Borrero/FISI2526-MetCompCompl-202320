import numpy as np

def polynomial_function(x):
    return 3.75*x**3 + 4.0*x**2 - 4.75*x + 5.0

def derivative_polynomial_function(x):
    return 11.25*x**2 + 8.0*x - 4.75

def newton_raphson_minimization(initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    
    for _ in range(max_iterations):
        x_next = x - derivative_polynomial_function(x) / (3 * 3.75 * x**2 + 2 * 4.0 * x - 4.75)
        
        if np.abs(x_next - x) < tolerance:
            break
        
        x = x_next
    
    return x, polynomial_function(x)

# Establecer un punto de inicio cercano al mínimo
initial_guess = 0.0

min_x, min_y = newton_raphson_minimization(initial_guess)

print("Punto mínimo:")
print("x:", min_x)
print("y:", min_y)

