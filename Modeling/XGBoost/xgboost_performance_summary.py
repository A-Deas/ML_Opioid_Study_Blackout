import pandas as pd 
import numpy as np 

input_path  = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/XGBoost/tuned_xgb_fold_metrics.csv"
output_path = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/XGBoost/xgboost_performance_summary.csv"

df = pd.read_csv(input_path)
df = df.drop(columns=["Fold"])

summary = pd.DataFrame({
    "Metric": df.columns,
    "Mean": df.mean().values, 
    "SD": df.std(ddof=1).values # std of the sample
})

summary["Mean +/- SD"] = summary.apply(
    lambda row: f"{row['Mean']:.4f} +/- {row['SD']:.4f}",
    axis=1
)

summary["Mean"] = summary["Mean"].round(4)
summary["SD"] = summary["SD"].round(4)

summary.to_csv(output_path, index=False)

print(summary)
print(f"\nSaved final performance summary.")