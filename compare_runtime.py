import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load Execution Times ===
with open("fno_runtime_single.txt", "r") as f:
    fno_runtime = float(f.readline().strip())

with open("numerical_runtime_single.txt", "r") as f:
    numerical_runtime = float(f.readline().strip())

# === Compute Speed-up Ratio ===
speedup_ratio = numerical_runtime / fno_runtime

# === Create a Comparison Table ===
runtime_data = pd.DataFrame({
    "Method": ["Numerical Solver", "FNO"],
    "Execution Time (s)": [numerical_runtime, fno_runtime],
    "Speed-up Ratio": [1.0, speedup_ratio]  # Numerical method is baseline
})

# Save as CSV for documentation
runtime_data.to_csv("runtime_comparison.csv", index=False)
print("Runtime comparison saved as 'runtime_comparison.csv'!")

# === Plot Bar Chart ===
plt.figure(figsize=(6, 4))
plt.bar(["Numerical Solver", "FNO"], [numerical_runtime, fno_runtime], color=["red", "blue"])
plt.xlabel("Method")
plt.ylabel("Execution Time (s)")
plt.title("Computational Efficiency Comparison")
plt.grid(axis="y")

# Save plot
plt.savefig("runtime_comparison.png")
plt.show()

print(f"Speed-up Ratio: {speedup_ratio:.2f}x faster!")
