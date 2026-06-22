import numpy as np 
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import duckdb
import pandas as pd
import json
from scipy.stats import gaussian_kde
pd.set_option("display.max_columns", None)

# --- Preliminaries
DB_PATH = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Tables/DuckDB_Files/ModelReady_FB.db"
MODEL_SAVE_DIR = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Saved_Models"
FI_SAVE_DIR = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/Feature_Importance/Folded_FI"

RANDOM_SEED = 42
NUM_FOLDS = 12

NON_FEATURE_COLS = {"PatientSID", "AOE_PatLabel"}

CATEGORICAL_COLS = [
    "IndexType",
    "BirthSex",
    "MaritalStatus",
    "Race",
    "Ethnicity",
    "MilitarySexualTrauma"
]

def load_data():
    print("\nLoading data from DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("select * from ModelReady_FB order by PatientSID").df()
    con.close()
    print(f"Loaded {len(df):,} patients.")
    return df

def prepare_features(df):
    # feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
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
    return X, y, feature_cols

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
    # C=(0.20, 0.40, 0.60, 0.80),
    # C=(0.50, 0.80, 0.95, 0.99),
    # C=(0.50, 0.65, 0.80, 0.90),
    # C=(0.25, 0.50, 0.75, 0.99),
    # C=(0.20, 0.60, 0.90, 0.99),
    C=(0.20, 0.60, 0.80, 0.99),
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

    return tuple(thresholds) # (t1, t2, t3, t4)

def classify_on_test_set(probs, labels, thresholds, test_sids):
    t1, t2, t3, t4 = thresholds
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    levels = np.ones_like(probs, dtype=int) * 1
    levels[probs >= t1] = 2 
    levels[probs >= t2] = 3 
    levels[probs >= t3] = 4
    levels[probs >= t4] = 5

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
                    t4 = ("t4", "max")) # save t2 for this fold
               .reset_index()
               .sort_values("Level")
    )

    print("\nRisk-level summary (on the test set):")
    for _, row in summary.iterrows():
        lvl = int(row.Level)
        print(f"Level {lvl}: total_patients = {int(row.total_patients):,} | num_pos = {int(row.num_pos):,} | pct_pos = {row.pct_pos:3f} | avg_pred_prob = {row.avg_pred_prob:.3f}")

    return risk_df, summary 

def main():
    df = load_data()
    X, y, feature_cols = prepare_features(df)

    sid_to_idx = {sid: i for i, sid in enumerate(df["PatientSID"].values)}

    folds = build_folds(df)

    all_summaries = []

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
        t1, t2, t3, t4 = thresholds

        # thresholds = float(0.50), float(0.80), float(0.95), float(0.99)
        # t1, t2, t3, t4 = thresholds

        print(f"\nt1 = {t1:.2f}, t2 = {t2:.2f}, t3 = {t3:.2f}, and t4 = {t4:.2f}")
        risk_df, summary = classify_on_test_set(test_probs, test_y, thresholds, test_sids)
        all_summaries.append(summary)

    # --- Combine and print average RC results across all folds
    combined = pd.concat(all_summaries, ignore_index=True)
    avg_summary = (
        combined.groupby("Level", as_index=False)
        .agg({
            "total_patients": "mean",
            "num_pos": "mean",
            "pct_pos": "mean",
            "avg_pred_prob": "mean",
            "t1": "mean",
            "t2": "mean",
            "t3": "mean",
            "t4": "mean"
        })
    )
    avg_summary["Level"] = avg_summary["Level"].apply(lambda x: f"Level {int(x)}")
    avg_summary = avg_summary.round(4)

    print("\n--- Average risk classification summary across all folds:")
    print(avg_summary[["Level", "total_patients", "num_pos", "pct_pos", "avg_pred_prob", "t1", "t2", "t3", "t4"]])
    avg_summary.to_csv("/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Risk_Classification/final_risk_summary.csv", index=False)

if __name__ == "__main__":
    main()