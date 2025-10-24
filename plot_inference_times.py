import sys
import pandas as pd
import matplotlib.pyplot as plt

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python plot_inference_times.py <csv_path>")
    sys.exit(1)

# Read CSV
csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# Check required columns
required_cols = {"Dataset", "Model", "MemSize", "NumGPUs", "GPU_Names", "InferenceTime_s"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")


# Collect dataset name
dataset_names = df["Dataset"].unique()
if len(dataset_names) == 1:
    dataset_name = dataset_names[0]
else:
    dataset_name = "Multiple_Datasets"

# Collect GPU info as a string
gpu_info_list = df[["NumGPUs", "GPU_Names"]].drop_duplicates()
gpu_info_text = ", ".join([f"{int(row['NumGPUs'])} GPU(s): {row['GPU_Names']}" 
                            for _, row in gpu_info_list.iterrows()])

# Plot setup
plt.figure(figsize=(9, 6))

# Plot lines per model
for model_name, subdf in df.groupby("Model"):
    subdf = subdf.sort_values("MemSize")
    plt.plot(subdf["MemSize"], subdf["InferenceTime_s"], marker="o", linewidth=2, label=model_name)

# Title with GPU info below
plt.suptitle(f"{dataset_name} Inference Time vs Memory Size")
plt.title(f"{gpu_info_text}")
# Labels, grid, legend
plt.xlabel("Memory Size")
plt.ylabel("Inference Time (s)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Model")
plt.tight_layout()

# Save plot
output_filename = f"plot_inference_times_{dataset_name}.png"
plt.savefig(output_filename, dpi=200)
print(f"Saved plot as {output_filename}")