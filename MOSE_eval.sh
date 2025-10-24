#!/bin/bash
python davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --davis_path ./MOSE \
    --results_path outputs/MOSE_pred_pngs/MOSE_sam2_hiera_base_plus_memsize7

python davis2017-evaluation/evaluation_method.py \
    --task semi-supervised \
    --davis_path ./MOSE \
    --results_path outputs/MOSE_pred_pngs/MOSE_sam2_hiera_large_memsize7