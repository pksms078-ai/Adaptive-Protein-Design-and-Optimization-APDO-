import numpy as np

# Electrostatic energy (Coulomb-like interaction)
def calculate_electrostatic_energy(structure):
    energy = 0.0
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            r = np.linalg.norm(structure[i] - structure[j]) + 1e-6
            energy += 1.0 / r
    return energy


# Van der Waals energy (Lennard-Jones potential)
def calculate_van_der_waals_energy(structure):
    energy = 0.0
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            r = np.linalg.norm(structure[i] - structure[j]) + 1e-6
            energy += (1 / r**12) - (2 / r**6)
    return energy


# Solvation energy (compactness penalty)
def calculate_solvation_energy(structure):
    center = np.mean(structure, axis=0)
    distances = np.linalg.norm(structure - center, axis=1)
    return np.sum(distances)


# Total APDO scoring function
def scoring_function(structure):
    electrostatic_energy = calculate_electrostatic_energy(structure)
    van_der_waals_energy = calculate_van_der_waals_energy(structure)
    solvation_energy = calculate_solvation_energy(structure)
    total_energy = electrostatic_energy + van_der_waals_energy + solvation_energy
    return total_energy
