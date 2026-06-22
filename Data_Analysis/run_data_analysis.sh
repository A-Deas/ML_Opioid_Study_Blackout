#!/bin/bash
#SBATCH --job-name=xgb_DA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=12
#SBATCH --time=100:00:00
#SBATCH --output=/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Analysis/log_DA.out
#SBATCH --error=/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Analysis/log_DA.err

echo "Job started on $(date)"

/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/.venv/bin/python -u /mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Analysis/data_analysis.py

echo "Job finished on $(date)"