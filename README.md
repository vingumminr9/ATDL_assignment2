# Running code on UCloud cluster

Open a job with a terminal.

---

## Installing Conda

Follow the guide: [https://docs.cloud.sdu.dk/hands-on/conda-setup.html](https://docs.cloud.sdu.dk/hands-on/conda-setup.html)  

Navigate to `/work` folder and run:

```bash
curl -s -L -o /tmp/miniconda_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash /tmp/miniconda_installer.sh -b -f -p /work/miniconda3

sudo ln -s /work/miniconda3/bin/conda /usr/bin/conda

conda init

conda update conda

conda create --name sam2 python=3.10

conda activate sam2
```

---

## SAM2 setup

Installing SAM2 and required packages from [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2):

```bash
pip3 install torch torchvision

mkdir ATDL

cd ATDL

git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

We then need to download the right checkpoints.  
To compare with the model in the paper, we need to change the checkpoints to be from SAM2, and not SAM2.1, which is done by editing the file:

```text
/work/ATDL/sam2/checkpoints/download_ckpts.sh
```

In this file, remove the comments from the SAM2 checkpoints, and comment out the SAM2.1 checkpoints.  
Now we download the checkpoints:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

---

## DAVIS and MOSE dataset setup

### DAVIS

We now download and unzip the DAVIS Semi-supervised dataset in 480p (used for metrics on the website for DAVIS dataset). SAM2 paper references that they follow official evaluation tools.

```bash
cd ..
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

unzip DAVIS-2017-trainval-480p.zip

rm DAVIS-2017-trainval-480p.zip
```

---

## Running Inference

To change the memory when running the script, modify `vos_inference.py`. After line 380, add:

```python
parser.add_argument(
    "--sam2_memsize",
    type=int,
    default=7,
    help="Memory size of SAM2 model (overrides config value)"
)
```

And after the predictor is defined in line 456, add:
```python
predictor.num_maskmem = args.sam2_memsize
```

We are now ready to perform Semi-supervised VOS inference.  
The script allows switching between DATASETS (DAVIS vs MOSE), SAM2 models (base_plus vs large), and the amount of frames stored in memory (num_maskmem = 0..7).  

This script also dynamically runs on multiple GPUs by checking available GPUs and splitting videos evenly between them (for larger datasets, splitting by size may be optimal).

To run inference on the DAVIS dataset using the `base_plus` model with memory size 7:

```bash
python run_model_script.py --dataset DAVIS --sam2_model base_plus --sam2_memsize 7
```

---

## Evaluation

Evaluation follows: [https://github.com/davisvideochallenge/davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)  

Download and install their code:

```bash
git clone https://github.com/davisvideochallenge/davis2017-evaluation.git && cd davis2017-evaluation

python setup.py install

pip install pandas

pip install opencv-python-headless

pip install scipy

pip install scikit-learn

pip install scikit-image

cd ..
```

Run evaluation on generated masks:

```bash
python davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --davis_path ./DAVIS \
    --results_path outputs/DAVIS_pred_pngs/DAVIS_sam2_hiera_base_plus_memsize7
```

Running time for `base_plus` on DAVIS, memsize=7: 41s, using 8 GPUs (CPU only).  
Results:

```text
J&F-Mean   J-Mean  J-Recall  J-Decay   F-Mean  F-Recall  F-Decay
0.885856 0.854672  0.921446 0.050835 0.917041  0.967444 0.072987
```

To plot inference times and J&F-means for DAVIS:

```bash
python plot_jf_means.py DAVIS

python plot_inference_times.py inference_times_DAVIS.csv
```

---

## MOSE

For the MOSE dataset:

Download and unzip `valid.tar.gz` and metadata:

```bash
mkdir MOSE

cd MOSE

pip install gdown

gdown 'https://drive.google.com/uc?id=1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR'

tar -xvzf valid.tar.gz

rm valid.tar.gz

gdown 'https://drive.google.com/uc?id=1MmhRXKXFnEwzaoeE0b_e_RQjvVLHOc7I'

cd ..
```

Convert MOSE metadata to DAVIS-like structure:

```bash
python convert_MOSE_json_to_txt.py
```

Run inference on MOSE using `base_plus` model with memory size 7:

```bash
python run_model_script.py --dataset MOSE --sam2_model base_plus --sam2_memsize 7
```

For MOSE evaluation, use their evaluation server: [https://codalab.lisn.upsaclay.fr/competitions/10703](https://codalab.lisn.upsaclay.fr/competitions/10703)
