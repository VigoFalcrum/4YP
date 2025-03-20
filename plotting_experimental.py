import re
import pandas as pd
import matplotlib.pyplot as plt

# Read overall_results.txt
with open('overall_results.txt', 'r') as f:
    data = f.read()

# Split the file into blocks by the delimiter (the line of dashes)
blocks = data.split("-------------------------------------------------------")
results = []

# Parse each block using regular expressions
for block in blocks:
    block = block.strip()
    if not block:
        continue

    # Extract configuration: max_depth, min_samples_split, and core count.
    config_match = re.search(
        r"Configuration:\s*max_depth=([^,]+),\s*min_samples_split=([^,]+),\s*Cores=([0-9]+)",
        block)
    if not config_match:
        continue
    max_depth = config_match.group(1).strip()
    min_samples_split = config_match.group(2).strip()
    cores = int(config_match.group(3).strip())

    # Extract iterations (if needed)
    iter_match = re.search(r"Iterations:\s*([0-9]+)", block)
    iterations = int(iter_match.group(1)) if iter_match else None

    # Extract latency mean and standard deviation
    latency_match = re.search(
        r"Latency:\s*Mean=([\d\.]+),\s*Std=([\d\.]+)",
        block)
    if not latency_match:
        continue
    mean_latency = float(latency_match.group(1))
    std_latency = float(latency_match.group(2))

    # Extract MoE (optional)
    moe_match = re.search(r"Latency MoE:\s*([\d\.]+)%", block)
    moe = float(moe_match.group(1)) if moe_match else None

    results.append({
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "cores": cores,
        "iterations": iterations,
        "mean_latency": mean_latency,
        "std_latency": std_latency,
        "moe": moe
    })

# Convert the results to a pandas DataFrame.
df = pd.DataFrame(results)

# Create a configuration label to group the configurations.
df['config'] = "max_depth=" + df['max_depth'] + ", min_samples_split=" + df['min_samples_split']

print("Extracted data:")
print(df)

# Now, plot mean latency versus core count for each configuration.
configs = df['config'].unique()
plt.figure(figsize=(10,6))
for conf in configs:
    subset = df[df['config'] == conf].sort_values('cores')
    plt.errorbar(subset['cores'], subset['mean_latency'], yerr=subset['std_latency'],
                 marker='o', capsize=5, label=conf)

plt.xlabel("Number of Cores")
plt.ylabel("Mean Latency (seconds)")
plt.title("DT Model Mean Latency vs. Number of Cores")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
