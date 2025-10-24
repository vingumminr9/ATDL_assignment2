Running code on UCloud cluster:

Open a job with a terminal.\\

Installing conda:\\
from \url{https://docs.cloud.sdu.dk/hands-on/conda-setup.html} \\
navigate to /work folder and run following:
\begin{verbatim}
    curl -s -L -o /tmp/miniconda_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    
    bash /tmp/miniconda_installer.sh -b -f -p /work/miniconda3

    sudo ln -s /work/miniconda3/bin/conda /usr/bin/conda

    conda init

    conda pdate conda

    conda create --name sam2 python=3.10

    conda activate sam2
\end{verbatim}
%
%
%
\subsection{SAM 2 setup}
Installing SAM 2 and required packages\\
from \url{https://github.com/facebookresearch/sam2}:
\begin{verbatim}
    pip3 install torch torchvision

    mkdir ATDL

    cd ATDL

    mkdir project_480p

    cd project_480p
    
    git clone https://github.com/facebookresearch/sam2.git && cd sam2

    pip install -e .
    
\end{verbatim}
%
%
%
We then need to download the right checkpoints.\\
To compare with the model in the paper, we need to change the checkpoints to be from SAM2, and not SAM2.1 which is done by editing the file:
\begin{verbatim}
    /work/ATDL/project_480p/sam2/checkpoints/download_ckpts.sh
\end{verbatim}
%
%
%
In this file, we remove the comments from the SAM2 checkpoints, and outcomment the SAM2.1 checkpoints.\\
Now we download the checkpoints:
\begin{verbatim}
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
\end{verbatim}
%
%
%
\subsection{DAVIS and MOSE dataset setup}
For the DAVIS dataset:\\
We now download and unzip the DAVIS Semi-supervised dataset, in 480p. (This is described as being the one used for metrics on the website for DAVIS dataset). And SAM2 paper references that they follow official evaluation tools.
\begin{verbatim}
    cd ..
    wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

    unzip DAVIS-2017-trainval-480p.zip

    rm DAVIS-2017-trainval-480p.zip
\end{verbatim}
%
%
%
For the MOSE dataset:\\
To download and unzip valid.tar.gz and metadata file run:
\begin{verbatim}
    mkdir MOSE

    cd MOSE
    
    pip install gdown
    
    gdown 'https://drive.google.com/uc?id=1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR'

    tar -xvzf valid.tar.gz

    rm valid.tar.gz

    gdown 'https://drive.google.com/uc?id=1MmhRXKXFnEwzaoeE0b_e_RQjvVLHOc7I'

    cd ..
\end{verbatim}
%
%
The MOSE dataset metadata is structured differently than DAVIS, so we convert it to similar structure by creating the following python file and running it:
\begin{verbatim}
    import json
    import os
    
    # Path to MOSE JSON metadata
    json_path = "MOSE/meta_valid.json"
    
    # Output TXT file compatible with SAM2
    output_txt = "MOSE/valid/ImageSets/val.txt"
    
    # Make sure output folder exists
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    # Load MOSE metadata
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract video IDs from the "videos" key
    video_ids = list(data["videos"].keys())
    
    # Write to TXT
    with open(output_txt, "w") as f:
        for vid in video_ids:
            f.write(f"{vid}\n")
    
    print(f"DAVIS-style TXT file created at: {output_txt}")
\end{verbatim}
%
%
%
\subsection{Running Inference}
We are now ready to perform Semi-supervised VOS inference.\\
In order to automate the process of running our ablation study, we create a script that allow us to change between DATASETS (DAVIS vs MOSE), Sam2 models (base\_plus vs large), and the amount of frames stored in memory (num\_maskmem = int).\\
%
This script also dynamically allow us to run on multiple GPU's by first checking the amount of available GPUs, and then splitting the videos evenly between them, since the inference part can be run parallel. (for larger datasets, it would be optimal to split them based on size, since some videos are larger than others).\\
%
In order to change the memory when running the script, we have to make a small change in the file vos\_inference.py. After line 380 we add:
%
\begin{verbatim}
    parser.add_argument(
    "--sam2_memsize",
    type=int,
    default=7,
    help="Memory size of SAM2 model (overrides config value)"
    )
\end{verbatim}
%
%
%
The script looks as follows:
%
%
%
\begin{verbatim}
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
        base_video_dir = "././MOSE/valid/JPEGImages"
        input_mask_dir = "././MOSE/valid/Annotations"
        video_list_file = "././MOSE/valid/ImageSets/val.txt"

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
\end{verbatim}
To run the inference on the DAVIS dataset, using the base\_plus model, with a memory size of 7, we run the following command:
\begin{verbatim}
    python run_model_script.py --dataset DAVIS --sam2_model base_plus --sam2_memsize 7
\end{verbatim}
%
%
Running time for base\_plus on DAVIS, with memsize = 7: 1m3s, using 8 GPUs.
Running time for large on DAVIS, with memsize = 7: 1m44s, using 8 GPUs.
%
%
%
%
%
To run the inference on the MOSE dataset, using the base\_plus model, with a memory size of 7, we run the following command:
\begin{verbatim}
    python run_model_script.py --dataset MOSE --sam2_model base_plus --sam2_memsize 7
\end{verbatim}
%
%
Running time for base\_plus on MOSE, with memsize = 7: ?m?s, using 8 GPUs.
Running time for large on MOSE, with memsize = 7: ?m?s, using 8 GPUs.
%
%
%
%
%
\subsection{Evaluation}
We run the evaluation as described on: \url{https://github.com/davisvideochallenge/davis2017-evaluation}\\
First we download and install their code:
\begin{verbatim}
    git clone https://github.com/davisvideochallenge/davis2017-evaluation.git && cd davis2017-evaluation
    
    python setup.py install

    pip install pandas

    pip install opencv-python-headless

    pip install scipy
    
    pip install scikit-learn
    
    pip install scikit-image
    
    cd ..
\end{verbatim}
%
%
%
To run the evaluation on the masks we just generated, we run:
\begin{verbatim}
    python davis2017-evaluation/evaluation_method.py --task semi-supervised --davis_path ./DAVIS --results_path outputs/DAVIS_pred_pngs/DAVIS_sam2_hiera_base_plus_memsize7
\end{verbatim}
%
%
Running time for base\_plus on DAVIS, with memsize = 7: 41s, using 8 GPUs (only CPU is utilized though).\\
results:
\begin{verbatim}
    J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
    0.885856 0.854672  0.921446 0.050835 0.917041  0.967444 0.072987
\end{verbatim}
%
%
%

%
%
And to run the evaluation on the large, we run:
\begin{verbatim}
    python davis2017-evaluation/evaluation_method.py --task semi-supervised --davis_path ./DAVIS --results_path outputs/DAVIS_pred_pngs/DAVIS_sam2_hiera_large_memsize7
\end{verbatim}
%%
%
Running time for large on DAVIS, with memsize = 7: 42s, using 8 GPUs (only CPU is utilized though).\\
results:
\begin{verbatim}
 J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
  0.89376 0.860923   0.92122 0.043284 0.926596  0.977366  0.05833
\end{verbatim}
%
%
%
%
%
%
%
%
%
Now we run our models on the MOSE dataset:
For the base-plus model on MOSE, run:
\begin{verbatim}
    python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2/sam2_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2_hiera_base_plus.pt \
  --base_video_dir ./MOSE/valid/JPEGImages \
  --input_mask_dir ./MOSE/valid/Annotations \
  --video_list_file ./MOSE/valid/ImageSets/val.txt \
  --output_mask_dir ./outputs/MOSE_predictions/base_plus
\end{verbatim}
%
%
%
First run took 38m6s (on 4 gpus, but only 1 used).
second run took 10m14s on (8 gpus, all utilized)
%
%
%

For the Large model, on MOSE run:
%
%
%
\begin{verbatim}
    python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2/sam2_hiera_l.yaml \
  --sam2_checkpoint ./checkpoints/sam2_hiera_large.pt \
  --base_video_dir ./MOSE/valid/JPEGImages \
  --input_mask_dir ./MOSE/valid/Annotations \
  --video_list_file ./MOSE/valid/ImageSets/val.txt \
  --output_mask_dir ./outputs/MOSE_predictions/large
\end{verbatim}
%
%
%
First run took 16m48s on (4 gpus with new script, utilizing all 4)
second run took 10m31s on (8 gpus). 
%
%
%

SAM 2 paper states that if a dataset has an official toolkit for evaluation then they use that, so we do the same.


------- NOTE - MOSE dataset only contains annotations for first frame in every video. \\
downloaded other file, called sample\_submission\_valid\_all...\\
MOSE dataset folders are rearranged and renamed to fit same structure as the DAVIS dataset.\\
Evaluation is run on them there. 