import numpy as np 
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import duckdb
import pandas as pd
import json
import os
from scipy.stats import gaussian_kde
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# --- Preliminaries
DB_PATH = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Tables/DuckDB_Files/ModelReadyBlackout.db"
MODEL_SAVE_DIR = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Saved_Models"
FI_SAVE_DIR = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/Feature_Importance/Folded_FI"

RANDOM_SEED = 42
NUM_FOLDS = 12 

BINARY_COLS = [ 
    'AlcoholUse_Ever', 'CannabisUse_Ever', 'CocaineUse_Ever', 'StimulantUse_Ever', 'HallucinogenUse_Ever', 'Depression_Ever', 'Anxiety_Ever', 'BipolarDisorder_Ever',
    'Schizophrenia_Ever', 'PTSD_Ever', 'ADHD_Ever', 'Obesity_Ever', 'Diabetes_Ever', 'COPD_Ever', 'Asthma_Ever', 'HeartFailure_Ever', 'Arthritis_Ever',
    'ChronicBackPain_Ever', 'PainNEC_Ever', 'MyofascialPain_Ever', 'EatingDisorder_Ever', 'FamilyDisruption_Ever', 'TobaccoUse_Ever', 'HepC_Ever', 'HIV_Ever'
]

NUMERIC_COLS = [ 
    'Age', 'NumVisits', 'TotalNumRx',
    'TotalDaysSupply_Min', 'TotalDaysSupply_Mean', 'TotalDaysSupply_Max',
    'MME_Min', 'MME_Mean', 'MME_Max',
    'NumFills_Min', 'NumFills_Mean', 'NumFills_Max',
    'NumRx_Oxycodone', 'NumRx_Fentanyl', 'NumRx_Meperidine', 'NumRx_Levorphanol', 'NumRx_Oxymorphone', 'NumRx_Hydrocodone',
    'NumRx_Morphine', 'NumRx_Tapentadol', 'NumRx_Tramadol', 'NumRx_Codeine', 'NumRx_Hydromorphone', 'NumRx_Buprenorphine', 'NumRx_Methadone',
] 
# 'AlcoholUse_Total', 'CannabisUse_Total', 'CocaineUse_Total', 'StimulantUse_Total', 'HallucinogenUse_Total', 'Depression_Total', 'Anxiety_Total', 
# 'BipolarDisorder_Total', 'Schizophrenia_Total', 'PTSD_Total', 'ADHD_Total', 'Obesity_Total', 'Diabetes_Total', 'COPD_Total', 'Asthma_Total',
# 'HeartFailure_Total', 'Arthritis_Total', 'ChronicBackPain_Total', 'PainNEC_Total', 'MyofascialPain_Total', 'EatingDisorder_Total',
# 'FamilyDisruption_Total', 'TobaccoUse_Total', 'HepC_Total', 'HIV_Total'

CATEGORICAL_COLS = [
    "IndexType",
    "BirthSex",
    "MaritalStatus",
    "Race",
    "Ethnicity",
    "MilitarySexualTrauma"
]

NON_FEATURE_COLS = {"PatientSID", "AOE_PatLabel"}

def load_data():
    print("\nLoading data from DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("select * from ModelReadyBlackout order by PatientSID").df()
    con.close()
    print(f"Loaded {len(df):,} patients.")
    return df

def prepare_features(df):
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and not c.endswith(("_Total", "_Q25", "_Median", "_Q75", "_Std"))
    ]

    X = df[feature_cols].copy()
    y = df["AOE_PatLabel"].astype(np.float32).values

    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    for col in feature_cols:
        if col not in CATEGORICAL_COLS and np.issubdtype(X[col].dtype, np.number):
            X[col] = X[col].astype(np.float32)

    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]:,} columns")
    return X, y

def build_folds(df):
    rng = np.random.default_rng(RANDOM_SEED)

    pos_sids = df.loc[df["AOE_PatLabel"] == 1, "PatientSID"].values 
    neg_sids = df.loc[df["AOE_PatLabel"] == 0, "PatientSID"].values 

    pos_sids = np.sort(pos_sids)
    neg_sids = np.sort(neg_sids)
    
    rng.shuffle(neg_sids)
    neg_folds = np.array_split(neg_sids, NUM_FOLDS)

    folds = [np.concatenate([neg_fold, pos_sids]) for neg_fold in neg_folds]

    n_pos = len(pos_sids)
    n_neg = len(neg_sids)
    return folds

def pick_tier_thresholds_kde(
    val_probs,
    val_labels,
    C=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
    bw_method="scott",
    grid_size=20001,
    eps=1e-6,
):
    """
    KDE-smoothed quantile thresholds based on positive-case predicted probabilities.

    Steps:
        - take probs for val_labels == 1 (positive cases)
        - logit-transform to R
        - fit Gaussian KDE in logit space
        - numerically integrate to get CDF on a grid
        - invert CDF to get thresholds at C
        - return thresholds in probability space
    """
    probs = np.asarray(val_probs, dtype=float)
    labels = np.asarray(val_labels, dtype=int)
    
    # Code chunk if you want to use ONLY positive patients when constructing thresholds
    # pos = probs[labels == 1] # probs for the positive patients
    # pos = np.clip(pos, eps, 1 - eps) # avoid infinities at 0/1
    # y = np.log(pos / (1 - pos)) # logit transform

    # Code chunk if you want to use ALL patients (both pos and neg) when constructing thresholds
    probs = np.clip(probs, eps, 1 - eps) # avoid infinities at 0/1
    y = np.log(probs / (1 - probs)) # logit transform

    kde = gaussian_kde(y, bw_method=bw_method)

    # Build a grid in logit space (cover tails)
    lo, hi = np.quantile(y, [0.001, 0.999])
    pad = 0.5 * (hi - lo)
    lo -= pad 
    hi += pad 
    grid = np.linspace(lo, hi, grid_size)

    # PDF on grid, then CDF by cumulative trapezoid 
    pdf = kde(grid)
    dx = grid[1] - grid[0]
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * dx)
    cdf = np.insert(cdf, 0, 0.0)
    cdf = cdf / cdf[-1] # normalize 

    # Invert CDF for desired quantiles
    thresholds = []
    for c in C:
        c = float(c)
        y_t = np.interp(c, cdf, grid) # inverse CDF via interpolation 
        p_t = 1 / (1 + np.exp(-y_t)) # logistic back-transform
        thresholds.append(float(p_t))

    return tuple(thresholds) # (t1, t2, t3, t4, ..., t10, t11)

def classify_on_test_set(probs, labels, thresholds, test_sids):
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = thresholds
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    levels = np.ones_like(probs, dtype=int) * 1
    levels[probs > t1] = 2 
    levels[probs > t2] = 3 
    levels[probs > t3] = 4
    levels[probs > t4] = 5
    levels[probs > t5] = 6
    levels[probs > t6] = 7
    levels[probs > t7] = 8
    levels[probs > t8] = 9
    levels[probs > t9] = 10
    levels[probs > t10] = 11
    levels[probs > t11] = 12

    test_sids = sorted(test_sids) # just to be super duper extra sure that all patients are ordered in exactly the same way in both the RC and DA

    risk_df = pd.DataFrame({
        "PatientSID": test_sids,
        "Prob": probs,
        "Label": labels,
        "Level": levels,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": t4,
        "t5": t5,
        "t6": t6,
        "t7": t7,
        "t8": t8,
        "t9": t9,
        "t10": t10,
        "t11": t11
    })

    sid_to_dummy = {sid: i for i, sid in enumerate(test_sids)}
    risk_df["DummyID"] = risk_df["PatientSID"].map(sid_to_dummy)
    risk_df = risk_df.drop(columns=["PatientSID"])

    # count negative patients per level (second entries in tuples across from equals signs are the aggregate functions in Pandas!)
    summary = (
        risk_df.groupby("Level") # get the following aggregate statistics for each risk level
               .agg(total_patients = ("DummyID", "size"), # total number of patients in that level
                    num_pos = ("Label", "sum"), # the number of positive patients in that level
                    pct_pos=("Label", "mean"), # mean of binary label = percentage of positive patients in that level
                    avg_pred_prob=("Prob", "mean"), # average predicted probability for that level,
                    t1 = ("t1", "max"), # save t1 for this fold
                    t2 = ("t2", "max"), # save t2 for this fold
                    t3 = ("t3", "max"), # save t3 for this fold
                    t4 = ("t4", "max"), # save t4 for this fold
                    t5 = ("t5", "max"), # save t5 for this fold
                    t6 = ("t6", "max"), # save t6 for this fold
                    t7 = ("t7", "max"), # save t7 for this fold
                    t8 = ("t8", "max"), # save t8 for this fold
                    t9 = ("t9", "max"), # save t9 for this fold
                    t10 = ("t10", "max"), # save t10 for this fold
                    t11 = ("t11", "max")) # save t11 for this fold
               .reset_index()
               .sort_values("Level")
    )

    print("\nRisk-level summary (on the test set):")
    for _, row in summary.iterrows():
        lvl = int(row.Level)
        print(f"Level {lvl}: total_patients = {int(row.total_patients):,} | num_pos = {int(row.num_pos):,} | pct_pos = {row.pct_pos:3f} | avg_pred_prob = {row.avg_pred_prob:.3f}")

    return risk_df, summary 

def summarize_by_risk_level(subset_name, test_X, risk_df, test_sids, fold_idx):
    print(f"\nRunning data analysis on the risk levels for {subset_name}...")

    test_X = test_X.copy()
    test_sids = sorted(test_sids) # just to be super duper extra sure that all patients are ordered in exactly the same way in both the RC and DA
    test_X["TestSID"] = test_sids

    sid_to_dummy = {sid: i for i, sid in enumerate(test_sids)}
    test_X["DummyID"] = test_X["TestSID"].map(sid_to_dummy)
    test_X = test_X.drop(columns=["TestSID"])

    if subset_name == "all":
        risk_df_subset = risk_df.copy()
    elif subset_name == "pos":
        risk_df_subset = risk_df[risk_df["Label"] == 1].copy()
    elif subset_name == "neg":
        risk_df_subset = risk_df[risk_df["Label"] == 0].copy()

    merged = pd.merge(risk_df_subset, test_X, on="DummyID", how="left")

    level_summaries = {}

    for level, subdf in merged.groupby("Level"):
        analysis = {}

        # Numeric feature summary
        analysis["numeric_summary"] = (
            subdf[NUMERIC_COLS]
            .agg(["mean"]) # skips NaNs
            .T
        )

        # Binary feature summary
        analysis["binary_summary"] = pd.DataFrame({
            # "count_1s": subdf[BINARY_COLS].sum(), # skips NaNs
            "proportion_1s": subdf[BINARY_COLS].mean() # skips NaNs
        })

        # Categorical feature summary
        cat_summary = {}
        for col in CATEGORICAL_COLS:
            cat_summary[col] = (
                subdf[col]
                .value_counts(normalize=True) # skips NaNs
                .rename("proportion")
                .to_frame()
            )
        analysis["categorical_summary"] = cat_summary
        analysis["num_patients"] = len(subdf)

        level_summaries[level]= analysis

    print("Data analysis complete.")
    return level_summaries

def print_level_summaries(avg_summaries, subset_name, out_dir="/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Analysis/Risk_Level_Summaries/"):
    level_names = {1: "Level 1", 2: "Level 2", 3: "Level 3", 4: "Level 4", 5: "Level 5", 6: "Level 6", 7: "Level 7", 8: "Level 8", 9: "Level 9", 10: "Level 10", 11: "Level 11", 12: "Level 12"}
    subset_pretty = {"all": "ALL", "pos": "POSITIVES", "neg": "NEGATIVES"}
    sep = "-" * 90

    for level, summary in avg_summaries.items():
        print(f"\n{sep}")
        print(f"{'-'*20} {level_names.get(level, f'{level}')} | {subset_pretty.get(subset_name, subset_name)}")
        print(f"{sep}")

        # Numeric summary
        print("\nNumeric summary:")
        numeric_sorted = (
            summary["numeric_summary"]
            .reset_index()
            .rename(columns = {"index": "Feature"})
            .round(3)
            .sort_values(by="mean", ascending=False)
        )
        numeric_sorted.to_csv(os.path.join(out_dir, f"numeric_summary_level_{level}_{subset_name}.csv"), index=False)
        with pd.option_context('display.max_rows', None):
            print(numeric_sorted)

        # Binary summary
        print("\nBinary summary:")
        binary_sorted = (
            summary["binary_summary"]
            .reset_index()
            .rename(columns = {"index": "Feature"})
            .round(3)
            .sort_values(by="proportion_1s", ascending=False)
        )
        binary_sorted.to_csv(os.path.join(out_dir, f"binary_summary_level_{level}_{subset_name}.csv"), index=False)
        print(binary_sorted) 

        # Categorical summary
        print("\nCategorical summary")
        for col, cat_df in summary["categorical_summary"].items():
            print(f"\n{col:}") 
            cat_sorted = cat_df.round(3).sort_values(by="avg_proportion", ascending=False)
            cat_sorted.to_csv(os.path.join(out_dir, f"{col}_summary_level_{level}_{subset_name}.csv"), index=False)
            print(cat_sorted)  

def average_summaries(summary_list):
    avg_summaries = {}

    for level in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        valid_summaries = [s[level] for s in summary_list if level in s]

        if len(valid_summaries) == 0:
            continue

        avg_summaries[level] = {}

        # Numeric
        numeric_frames = [s["numeric_summary"] for s in valid_summaries]
        avg_summaries[level]["numeric_summary"] = sum(numeric_frames) / len(numeric_frames)

        # Binary
        binary_frames = [s["binary_summary"] for s in valid_summaries]
        avg_summaries[level]["binary_summary"] = sum(binary_frames) / len(binary_frames)

        # Categorical
        avg_cat_summaries = {}
        cat_vars = set().union(*(s["categorical_summary"].keys() for s in valid_summaries))

        for var in cat_vars:
            dfs = []
            for s in valid_summaries:
                df = s["categorical_summary"].get(var)
                dfs.append(df)

            if len(dfs) == 0:
                print("problem")

            merged = pd.concat(dfs, axis=1).fillna(0)
            merged["avg_proportion"] = merged.mean(axis=1)
            
            avg_cat_summaries[var] = merged[["avg_proportion"]].sort_values(by="avg_proportion", ascending=False)

        avg_summaries[level]["categorical_summary"] = avg_cat_summaries

    return avg_summaries

def main():
    df = load_data()
    X, y = prepare_features(df)

    sid_to_idx = {sid: i for i, sid in enumerate(df["PatientSID"].values)}

    folds = build_folds(df)

    # Store summaries separately for each subset across folds
    all_subset_summaries = {
        "all": [],
        "pos": [],
        "neg": [],
    }

    for fold_idx, fold_sids in enumerate(folds, start=1):
        print(f"\n----------------- Fold {fold_idx}/{NUM_FOLDS} -----------------")
        
        idxs = [sid_to_idx[sid] for sid in fold_sids]
        fold_X = X.iloc[idxs].reset_index(drop=True)
        fold_y = y[idxs]

        train_X, temp_X, train_y, temp_y = train_test_split(
            fold_X, fold_y, 
            test_size=0.3, 
            random_state=RANDOM_SEED+fold_idx, 
            stratify=fold_y
        )

        val_X, test_X, val_y, test_y = train_test_split(
            temp_X, temp_y, 
            test_size=0.5, 
            random_state=RANDOM_SEED+fold_idx, 
            stratify=temp_y
        ) 

        # split the SIDS into training, validation, and test sets
        train_sids, temp_sids, train_y, temp_y = train_test_split(
            fold_sids, fold_y, 
            test_size=0.3, 
            random_state=42+fold_idx, 
            stratify=fold_y
        )

        val_sids, test_sids, val_y, test_y = train_test_split(
            temp_sids, temp_y, 
            test_size=0.5, 
            random_state=42+fold_idx, 
            stratify=temp_y
        )

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(f"/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Saved_Models/trained_xgb_fold_{fold_idx}.json")

        val_probs  = xgb_model.predict_proba(val_X)[:, 1]
        test_probs = xgb_model.predict_proba(test_X)[:, 1]

        thresholds = pick_tier_thresholds_kde(val_probs, val_y)
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = thresholds
        print(f"\nt1 = {t1:.2f}, t2 = {t2:.2f}, t3 = {t3:.2f}, t4 = {t4:.2f}, t5 = {t5:.2f}, t6 = {t6:.2f}, t7 = {t7:.2f}, t8 = {t8:.2f}, t9 = {t9:.2f}, t10 = {t10:.2f}, t11 = {t11:.2f}")

        risk_df, _ = classify_on_test_set(test_probs, test_y, thresholds, test_sids)

        for subset_name in ["all", "pos", "neg"]:
            level_summaries = summarize_by_risk_level(subset_name, test_X, risk_df, test_sids, fold_idx)
            all_subset_summaries[subset_name].append(level_summaries)

    for subset_name in ["all", "pos", "neg"]:
        avg_summaries = average_summaries(all_subset_summaries[subset_name])
        print_level_summaries(avg_summaries, subset_name=subset_name)

if __name__ == "__main__":
    main()