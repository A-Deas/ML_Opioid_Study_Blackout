import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import seaborn as sns 
from pathlib import Path 

TOP_N = 30

FEATURE_NAME_MAP = {

    # Demographics
    "IndexType": "Initial opioid type",
    "BirthSex": "Birth sex",
    "Age": "Age",
    "MaritalStatus": "Marital status",
    "Race": "Race",
    "Ethnicity": "Ethnicity",
    "MilitarySexualTrauma": "Military sexual trauma",

    # Aggregate prescription features
    "TotalNumRx": "Total number of prescriptions",
    "TotalDaysSupply_Min": "Minimum days supply",
    "TotalDaysSupply_Mean": "Average days supply",
    "TotalDaysSupply_Max": "Maximum days supply",

    "MME_Min": "Minimum MME",
    "MME_Mean": "Average MME",
    "MME_Max": "Maximum MME",

    "NumFills_Min": "Minimum fills",
    "NumFills_Mean": "Average fills",
    "NumFills_Max": "Maximum fills",

    # Specific opioid counts
    "NumRx_Oxycodone": "Number of oxycodone prescriptions",
    "NumRx_Fentanyl": "Number of fentanyl prescriptions",
    "NumRx_Meperidine": "Number of meperidine prescriptions",
    "NumRx_Levorphanol": "Number of levorphanol prescriptions",
    "NumRx_Oxymorphone": "Number of oxymorphone prescriptions",
    "NumRx_Hydrocodone": "Number of hydrocodone prescriptions",
    "NumRx_Morphine": "Number of morphine prescriptions",
    "NumRx_Tapentadol": "Number of tapentadol prescriptions",
    "NumRx_Tramadol": "Number of tramadol prescriptions",
    "NumRx_Codeine": "Number of codeine prescriptions",
    "NumRx_Hydromorphone": "Number of hydromorphone prescriptions",
    "NumRx_Buprenorphine": "Number of buprenorphine prescriptions",
    "NumRx_Methadone": "Number of methadone prescriptions",

    # Healthcare utilization
    "NumVisits": "Total number of encounters",

    # Substance use
    "AlcoholUse_Ever": "Alcohol use",
    "CannabisUse_Ever": "Cannabis use",
    "CocaineUse_Ever": "Cocaine use",
    "StimulantUse_Ever": "Stimulant use",
    "HallucinogenUse_Ever": "Hallucinogen use",
    "TobaccoUse_Ever": "Tobacco use",

    # Mental health
    "Depression_Ever": "Depression",
    "Anxiety_Ever": "Anxiety",
    "BipolarDisorder_Ever": "Bipolar disorder",
    "Schizophrenia_Ever": "Schizophrenia",
    "PTSD_Ever": "PTSD",
    "ADHD_Ever": "ADHD",
    "EatingDisorder_Ever": "Eating disorder",
    "FamilyDisruption_Ever": "Family disruption",

    # Medical conditions
    "Obesity_Ever": "Obesity",
    "Diabetes_Ever": "Diabetes",
    "COPD_Ever": "COPD",
    "Asthma_Ever": "Asthma",
    "HeartFailure_Ever": "Heart failure",
    "Arthritis_Ever": "Arthritis",

    # Pain conditions
    "ChronicBackPain_Ever": "Chronic back pain",
    "PainNEC_Ever": "Pain (not elsewhere classified)",
    "MyofascialPain_Ever": "Myofascial pain",

    # Infectious disease
    "HepC_Ever": "Hepatitis C",
    "HIV_Ever": "HIV"
}

def categorize_feature(name):
    if name.endswith("_Ever") or name.endswith("_Total") or name =="NumVisits":
        return "encounter", "darkorchid"

    if any(stat in name for stat in ["Min", "Q25", "Median", "Mean", "Q75", "Max", "Std", "IndexType", "TotalNumRx"]) or name.startswith("NumRx_"):
        return "rx", "darkorange"

    return "demo", "gray"

df = pd.read_csv("/Users/p5d/Documents/Python/ML_Opioid_Study_Blackout/Modeling/Feature_Importance/Folded_FI/xgb_FI_averaged.csv")
df = df[["feature", "mean_importance"]]
df = df.sort_values("mean_importance", ascending=False).head(TOP_N)

df["category"], df["color"] = zip(*df["feature"].apply(categorize_feature))
df["pretty_feature"] = df["feature"].map(FEATURE_NAME_MAP).fillna(df["feature"])

plt.figure(figsize=(10, max(6, 0.4 * len(df))))
sns.barplot(
    data=df,
    x="mean_importance",
    y="pretty_feature",
    palette=df["color"].tolist()
)

category_colors = {
    "Encounter": "darkorchid",
    "Prescription": "darkorange",
    "Demographic": "gray"
}

legend_handles = [
    mpatches.Patch(color=color, label=cat) for cat, color in category_colors.items()
]

plt.title("Feature importance", fontsize=20, weight="bold")
plt.xlabel("Importance score", fontsize=20, weight="bold")
plt.ylabel("Feature", fontsize=20, weight="bold")
plt.yticks(fontsize=16)
plt.xticks(fontsize=14)

legend = plt.legend(
    handles=legend_handles,
    title="Feature type",
    loc="lower right"
)

for text in legend.get_texts():
    text.set_fontsize(16)

legend.get_title().set_fontsize(20)
legend.get_title().set_fontweight("bold")


plt.tight_layout() 
plt.savefig("/Users/p5d/Documents/Python/ML_Opioid_Study_Blackout/Plots/FI/FI_raw.png", dpi=300)
plt.close()