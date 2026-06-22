import pandas as pd 
import glob 
import re 
from pathlib import Path 

DATA_DIR = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Analysis/Risk_Level_Summaries"
OUTPUT_CSV = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Analysis/DA_summary.csv"

LOW_LEVEL = 1
HIGH_LEVEL = 12

VALUE_COLUMNS = ["mean", "proportion_1s", "avg_proportion"]

def detect_value_column(df):
    for col in VALUE_COLUMNS:
        if col in df.columns:
            return col

def extract_level(filename):
    m = re.search(r"_level_(\d+)", filename)
    return int(m.group(1)) if m else None 

def detect_variable_type(filename):
    name = filename.lower()
    if "numeric" in name:
        return "numeric"
    if "binary" in name:
        return "binary"
    return "categorical"

def extract_category_name(filename):
    base = Path(filename).stem 
    category = base.split("_summary")[0]
    category = re.sub(r"_level_\d+", "", category)
    return category 

records = []

for filepath in glob.glob(f"{DATA_DIR}/*_summary_level_*_all.csv"):
    filename = Path(filepath).name 
    level = extract_level(filename)
    var_type = detect_variable_type(filename)

    if var_type != "categorical":
        category = extract_category_name(filename) 

        df = pd.read_csv(filepath)

        feature_col = df.columns[0]
        value_col = detect_value_column(df)

        for _, row in df.iterrows():
            feature_name = row[feature_col]

            if not feature_name.startswith("NumRx_"):
                value = row[value_col]

                records.append({
                    "variable": feature_name,
                    "category": category,
                    "variable_type": var_type,
                    "risk_level": level,
                    "value": value,
                })

df_all = pd.DataFrame(records)
# print(df_all)

df_pivot = (
    df_all[df_all["risk_level"].isin([LOW_LEVEL, HIGH_LEVEL])]
    .pivot_table(index=["variable", "variable_type", "category"],
                 columns="risk_level",
                 values="value")
    .reset_index()
)

df_pivot = df_pivot.rename(columns={
    LOW_LEVEL: "Level 1",
    HIGH_LEVEL: "Level 12"
})

df_pivot["percent_diff"] = 100 * (
    (df_pivot["Level 12"] - df_pivot["Level 1"]) / 
    ((df_pivot["Level 12"] + df_pivot["Level 1"]) / 2)
)
df_final = df_pivot.sort_values("percent_diff", ascending=False)

mask_non_numeric = df_final["variable_type"] != "numeric"

df_final.loc[mask_non_numeric, "Level 1"] = df_final.loc[mask_non_numeric, "Level 1"] * 100
df_final.loc[mask_non_numeric, "Level 12"] = df_final.loc[mask_non_numeric, "Level 12"] * 100

df_final["Level 1"] = df_final["Level 1"].round(4)
df_final["Level 12"] = df_final["Level 12"].round(4)
df_final["percent_diff"] = df_final["percent_diff"].round(4)

df_final.to_csv(OUTPUT_CSV, index=False)

print("\n DA summary saved to csv.")
# print("\n--- Final combined table ---")
# with pd.option_context('display.max_columns', None):
#     print(df_final.head(50))