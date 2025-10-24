import os
import csv

datasets = ["DAVIS", "MOSE"]
models = ["base_plus", "large"]
memsizes = range(0, 8)  # inclusive 0–7

output_file = "combined_jfmean_results.csv"

# Write header
with open(output_file, "w", newline="") as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(["Dataset", "Model", "MemSize", "J&F-Mean"])

    for dataset in datasets:
        for model in models:
            for memsize in memsizes:
                csv_path = f"outputs/{dataset}_pred_pngs/{dataset}_sam2_hiera_{model}_memsize{memsize}/global_results-val.csv"

                if not os.path.exists(csv_path):
                    print(f"⚠️ Missing: {csv_path}")
                    continue

                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    row = next(reader, None)
                    if row is None:
                        print(f"⚠️ Empty file: {csv_path}")
                        continue

                    jf_mean = row.get("J&F-Mean")
                    writer.writerow([dataset, model, memsize, jf_mean])

print(f"✅ Combined results saved to {output_file}")
