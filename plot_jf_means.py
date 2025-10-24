import os
import sys
import csv
import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend for clusters
import matplotlib.pyplot as plt

# --- Usage check ---
if len(sys.argv) < 2:
    print("Usage: python plot_jfmeans.py <dataset_name>")
    sys.exit(1)

dataset = sys.argv[1]  # e.g. DAVIS or MOSE
print(f"Generating J&F plot for {dataset}")

# --- Configuration ---
models = ["base_plus", "large"]
memsizes = range(0, 8)
base_path = f"outputs/{dataset}_pred_pngs"
results = {}

# --- Load results ---
for model in models:
    jf_means = []
    for mem in memsizes:
        result_dir = f"{base_path}/{dataset}_sam2_hiera_{model}_memsize{mem}"
        csv_path = os.path.join(result_dir, "global_results-val.csv")

        if not os.path.exists(csv_path):
            print(f"Missing: {csv_path}")
            jf_means.append(None)
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)  # only one line expected
            jf_mean = float(row["J&F-Mean"])
            jf_means.append(jf_mean)

    results[model] = jf_means

# --- Plot ---
plt.figure(figsize=(8, 5))
for model, jf_means in results.items():
    valid_x = [m for m, v in zip(memsizes, jf_means) if v is not None]
    valid_y = [v for v in jf_means if v is not None]
    plt.plot(valid_x, valid_y, marker='o', label=model)

plt.xlabel("Memory Size")
plt.ylabel("J&F Mean")
plt.title(f"{dataset} Performance vs Memory Size")
plt.legend(title="Model")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# --- Save the plot ---
output_path = f"jfmean_vs_memsize_{dataset}.png"
plt.savefig(output_path, dpi=200)
print(f"Plot saved to {output_path}")
