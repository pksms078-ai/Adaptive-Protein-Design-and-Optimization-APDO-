import numpy as np
from .energy import scoring_function


# Generate a new structure by perturbing the current structure
def perturb_structure(structure, step_size=0.1):
    new_structure = structure.copy()
    index = np.random.randint(len(structure))
    new_structure[index] += np.random.normal(0, step_size, size=3)
    return new_structure


# Adaptive optimization algorithm
def adaptive_optimization(initial_structure, max_iterations):
    current_structure = initial_structure
    current_energy = scoring_function(current_structure)

    for iteration in range(max_iterations):
        new_structure = perturb_structure(current_structure)
        new_energy = scoring_function(new_structure)

        # Accept if energy improves
        if new_energy < current_energy:
            current_structure = new_structure
            current_energy = new_energy

    return current_structure
