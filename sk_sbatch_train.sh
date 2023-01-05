#!/bin/bash
#SBATCH --job-name=elizabethgooch_thesis_run
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=primary
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs_hamming/train-out-%j.txt

. /etc/profile

module load lang/miniconda3/4.10.3

source activate pytorch_audio

python sklearn_train.py
# python sklearn_evaluate.py
# python sklearn_synthesize_results.py
# --model_dir="terminal_output_$(echo $USER)_$(date +%Y-%m-%d_%H-%M-%S-%N)/" \
