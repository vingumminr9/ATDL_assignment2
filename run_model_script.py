import os
import subprocess
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["MOSE", "DAVIS"], required=True,
                        help="Dataset to run inference on (MOSE or DAVIS)")
    
    parser.add_argument("--sam2_model", type=str, choices=["base_plus", "large"], required=True,
                        help="Sam2 model to run inference on (base_plus or large)")
    
    parser.add_argument("--sam2_memsize", type=int, default=7, required=False,
                        help="Sam2 memory size (int)")
    
    parser.add_argument("--extra_flags", nargs="*", default=[],
                        help="Additional flags to pass to vos_inference.py, e.g. --use_all_masks")
    
    return parser.parse_args()

def get_dataset_paths(dataset):
    if dataset == "MOSE":
        base_video_dir = "././MOSE/JPEGImages/480p"
        input_mask_dir = "././MOSE/Annotations/480p"
        video_list_file = "././MOSE/ImageSets/2017/val.txt"

    elif dataset == "DAVIS":
        base_video_dir = "././DAVIS/JPEGImages/480p"
        input_mask_dir = "././DAVIS/Annotations/480p"
        video_list_file = "././DAVIS/ImageSets/2017/val.txt"

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return base_video_dir, input_mask_dir, video_list_file

def get_sam_config_and_checkpoint(sam2_model):
    if sam2_model == "base_plus":
        sam2_config = 'configs/sam2/sam2_hiera_b+.yaml'
        sam2_checkpoint = 'sam2/checkpoints/sam2_hiera_base_plus.pt'

    elif sam2_model == "large":
        sam2_config = 'configs/sam2/sam2_hiera_l.yaml'
        sam2_checkpoint = 'sam2/checkpoints/sam2_hiera_large.pt'

    else:
        raise ValueError(f"Unknown SAM2 model: {sam2_model}")
    return sam2_config, sam2_checkpoint

def main():
    args = parse_args()
    base_video_dir, input_mask_dir, video_list_file = get_dataset_paths(args.dataset)

    sam2_config, sam2_checkpoint = get_sam_config_and_checkpoint(args.sam2_model)

    # Generate output folder based on dataset and checkpoint filename and memory size.
    checkpoint_name = os.path.splitext(os.path.basename(sam2_checkpoint))[0]
    output_mask_dir = os.path.join("./outputs", f"{args.dataset}_pred_pngs", f"{args.dataset}_{checkpoint_name}_memsize{args.sam2_memsize}")
    os.makedirs(output_mask_dir, exist_ok=True)

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    # Read all videos
    with open(video_list_file, "r") as f:
        video_names = [v.strip() for v in f.readlines()]

    # Split video names across GPUs
    split_video_lists = [[] for _ in range(num_gpus)]
    for i, vid in enumerate(video_names):
        split_video_lists[i % num_gpus].append(vid)

    # Save temporary split TXT files
    split_txt_files = []
    for gpu_idx, vids in enumerate(split_video_lists):
        split_txt = f"./temp_val_part_gpu{gpu_idx}.txt"
        with open(split_txt, "w") as f:
            for v in vids:
                f.write(v + "\n")
        split_txt_files.append(split_txt)

    # Launch inference processes
    vos_script = "sam2/tools/vos_inference.py"
    processes = []
    for gpu_idx, split_txt in enumerate(split_txt_files):
        cmd = [
            "python", vos_script,
            "--sam2_cfg", sam2_config,
            "--sam2_checkpoint", sam2_checkpoint,
            "--base_video_dir", base_video_dir,
            "--input_mask_dir", input_mask_dir,
            "--video_list_file", split_txt,
            "--output_mask_dir", output_mask_dir,
            "--sam2_memsize", str(args.sam2_memsize)
        ] + args.extra_flags

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        print(f"[INFO] Launching inference on GPU {gpu_idx} with {len(split_video_lists[gpu_idx])} videos...")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.wait()

    print(f"[INFO] All inference completed. Output saved to {output_mask_dir}")

    # --- Cleanup temporary files ---
    for temp_file in split_txt_files:
        try:
            os.remove(temp_file)
            print(f"[INFO] Removed temporary file: {temp_file}")
        except OSError as e:
            print(f"[WARNING] Could not remove temp file {temp_file}: {e}")


if __name__ == "__main__":
    main()
