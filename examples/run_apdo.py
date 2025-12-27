from apdo.core import design_protein
from apdo.energy import scoring_function

# Test protein sequence
sequence = "MGKGSSVQPYNRCKGTFALPNYVDKVRG"

# Run APDO protein design
optimized_structure = design_protein(sequence)

# Output results
print("Optimized Protein Structure:")
print(optimized_structure)

print("\nFinal APDO Energy:")
print(scoring_function(optimized_structure))
