#!/bin/bash
set -euo pipefail

echo "===== Starting Inferencec experiments ====="

# Function to get GPU info
get_gpu_info() {
  # Number of GPUs
  num_gpus=$(nvidia-smi -L | wc -l)
  # GPU names (comma-separated)
  gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
  echo "$num_gpus,$gpu_names"
}

# Function to run and time inference
run_and_time() {
  local dataset=$1
  local model=$2
  local memsize=$3

  echo ">>> Running $model model on $dataset with memsize=${memsize}"

  # Record start time
  start=$(date +%s.%N)

  python run_model_script.py --dataset "$dataset" --sam2_model "$model" --sam2_memsize "$memsize"

  # Record end time
  end=$(date +%s.%N)
  runtime=$(echo "$end - $start" | bc)

  # Get GPU info
  IFS=',' read -r num_gpus gpu_names <<< "$(get_gpu_info)"

  echo ">>> Inference completed in ${runtime}s on ${num_gpus} GPU(s): ${gpu_names}"

  echo ">>> Evaluating $model results (memsize=${memsize}) on $dataset"
  python davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --davis_path "./$dataset" \
    --results_path "outputs/${dataset}_pred_pngs/${dataset}_sam2_hiera_${model}_memsize${memsize}"

  # Append timing + GPU info to CSV
  echo "${dataset},${model},${memsize},${num_gpus},\"${gpu_names}\",${runtime}" >> "$OUTFILE"
}

# --- Run for base_plus and large model ---
for dataset in MOSE DAVIS; do

  # Create per-dataset CSV
  OUTFILE="inference_times_${dataset}.csv"
  echo "Dataset,Model,MemSize,NumGPUs,GPU_Names,InferenceTime_s" > "$OUTFILE"
  
  for model in base_plus large; do
    for i in {0..7}; do
      run_and_time "$dataset" "$model" "$i"
    done
  done
  
# Create a plot for this dataset
  echo ">>> Plotting results for $dataset"
  python plot_inference_times.py "$OUTFILE"
  python plot_jf_means.py "$dataset"
done


echo "===== All experiments complete! ====="