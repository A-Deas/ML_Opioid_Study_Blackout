import numpy as np 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import xgboost as xgb
import duckdb
import pandas as pd
import json
import matplotlib.pyplot as plt 

# --- Preliminaries
DB_PATH = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Tables/DuckDB_Files/ModelReadyBlackout.db"
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
    df = con.execute("select * from ModelReadyBlackout order by PatientSID").df()
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

def plot_auc_curves(y_trues, y_probs):

    # Plot the ROC curve
    fpr_grid = np.linspace(0, 1, 200)
    tprs = []

    for i in range(NUM_FOLDS):
        fpr, tpr, _ = roc_curve(y_trues[i], y_probs[i])
        auc_val = auc(fpr, tpr)

        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(fpr_grid, mean_tpr)
    mean_std_tpr = std_tpr.mean()

    plt.figure(figsize=(8,6))
    plt.plot(fpr_grid, mean_tpr, color="black", lw=1.5, label=f"AUC-ROC = 0.8500")
    plt.fill_between(fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, color="gray", alpha=0.2, label=f"+/- 0.0028 (SD)")
    plt.plot([0,1], [0,1], lw=1.5, linestyle="--", color="gray")

    plt.xlabel("False positive rate", fontsize=20, weight="bold")
    plt.ylabel("True positive rate", fontsize=20, weight="bold")
    plt.title("AUC-ROC curve", fontsize=20, weight="bold")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    legend = plt.legend(loc="lower right")
    
    for text in legend.get_texts():
        text.set_fontsize(16)
    
    plt.tight_layout()
    plt.savefig(f"/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Plots/AUC/auc_roc.png", dpi=300)
    plt.close()

    # Plot the PR curve
    recall_grid = np.linspace(0, 1, 200)
    precisions = []
    aps = []
    baselines = []

    for i in range(NUM_FOLDS):
        precision, recall, _ = precision_recall_curve(y_trues[i], y_probs[i])
        ap = average_precision_score(y_trues[i], y_probs[i])
        aps.append(ap)

        interp_prec = np.interp(recall_grid, recall[::-1], precision[::-1])
        precisions.append(interp_prec)

        baselines.append(y_trues[i].mean())

    mean_prec = np.mean(precisions, axis=0)
    std_prec = np.std(precisions, axis=0)
    mean_ap = np.mean(aps)
    mean_std_prec = std_prec.mean()

    plt.figure(figsize=(8,6))
    plt.plot(recall_grid, mean_prec, color="black", lw=1.5, label=f"AUC-PR = 0.4251")
    plt.fill_between(recall_grid, mean_prec - std_prec, mean_prec + std_prec, color="gray", alpha=0.2, label=f"+/- 0.0092 (SD)")

    mean_baseline = np.mean(baselines)
    plt.axhline(mean_baseline, color="gray", linestyle="--", linewidth=1.5, label=f"Baseline = {mean_baseline:.4f}")

    plt.xlabel("Recall", fontsize=20, weight="bold")
    plt.ylabel("Precision", fontsize=20, weight="bold")
    plt.title("AUC-PR curve", fontsize=20, weight="bold")
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    legend = plt.legend(loc="upper right")
    
    for text in legend.get_texts():
        text.set_fontsize(16)

    plt.tight_layout()
    plt.savefig(f"/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Plots/AUC/auc_pr.png", dpi=300)
    plt.close()

def main():
    df = load_data()
    X, y, feature_cols = prepare_features(df)

    sid_to_idx = {sid: i for i, sid in enumerate(df["PatientSID"].values)}

    folds = build_folds(df)

    y_trues = []
    y_probs = []

    for fold_idx, fold_sids in enumerate(folds, start=1):
        print(f"\nGrabbing labels and predicted probabilities from fold {fold_idx}/{NUM_FOLDS}...")
        
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

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(f"/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Saved_Models/trained_xgb_fold_{fold_idx}.json")

        test_probs = xgb_model.predict_proba(test_X)[:, 1]

        y_trues_fold = test_y
        y_probs_fold = test_probs

        y_trues.append(y_trues_fold)
        y_probs.append(y_probs_fold)

    plot_auc_curves(y_trues, y_probs)

if __name__ == "__main__":
    main()

