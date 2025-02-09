import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["Numerical Solver", "FNO"]
execution_times = [3.29, 0.0127]  # Measured times
speedup_ratio = [1, 258.98]

# Plot
plt.figure(figsize=(6, 4))
plt.bar(methods, execution_times, color=["red", "blue"])
plt.yscale("log")  # Log scale for better visualization
plt.ylabel("Execution Time (seconds, log scale)")
plt.title("Computational Efficiency Comparison")

# Save and Show
plt.savefig("runtime_comparison.png")
plt.show()
