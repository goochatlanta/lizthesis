#!/bin/bash
#SBATCH --job-name=elizabethgooch_thesis_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=beards
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs_hamming/build-out-%j.txt

. /etc/profile

module load lang/miniconda3/4.10.3

source activate pytorch_audio

# python data/convert_2_wav.py
# python data/conversion_pooled.py - this did not work
# python data/annotation.py
python build_dataset.py
# python model/data_loader.py