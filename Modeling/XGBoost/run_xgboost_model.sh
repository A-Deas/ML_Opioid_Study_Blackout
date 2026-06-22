#!/bin/bash
#SBATCH --job-name=xgb_model
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --output=/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/XGBoost/Logs/log_xgb_model.out
#SBATCH --error=/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/XGBoost/Logs/log_xgb_model.err

echo "Job started on $(date)"

/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/.venv/bin/python -u /mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/XGBoost/xgboost_model.py

echo "Job finished on $(date)"