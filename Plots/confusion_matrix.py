import numpy as np 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import xgboost as xgb
import duckdb
import pandas as pd
import json
pd.set_option("display.max_columns", None)

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

def main():
    df = load_data()
    X, y, feature_cols = prepare_features(df)

    sid_to_idx = {sid: i for i, sid in enumerate(df["PatientSID"].values)}

    folds = build_folds(df)

    fold_confusions = []
    fold_thresh_list = [0.91, 0.92, 0.90, 0.92, 0.92, 0.92, 0.91, 0.92, 0.91, 0.92, 0.91, 0.91]

    for fold_idx, fold_sids in enumerate(folds, start=1):
        print(f"\nGrabbing labels and predictions from fold {fold_idx}/{NUM_FOLDS}...")
        
        idxs = [sid_to_idx[sid] for sid in fold_sids]
        fold_X = X.iloc[idxs].reset_index(drop=True)
        fold_y = y[idxs]
        fold_thresh = fold_thresh_list[fold_idx - 1]

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
        test_preds = (test_probs >= fold_thresh).astype(int)

        cm = confusion_matrix(test_y, test_preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()

        fold_confusions.append({"TN": tn, "FP": fp, "FN": fn, "TP": tp})

    # Average CM across folds
    conf_df = pd.DataFrame(fold_confusions)
    avg_counts = conf_df.mean().round(0).astype(int)

    avg_cm = np.array([
        [avg_counts["TP"], avg_counts["FN"]],
        [avg_counts["FP"], avg_counts["TN"]]
    ])

    avg_cm_df = pd.DataFrame(
        avg_cm,
        index=["True 1", "True 0"],
        columns=["Pred 1", "Pred 0"]
    )

    avg_cm_df.to_csv("/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Plots/CM/confusion_matrix.csv", index=True)
    print(avg_cm_df)
    print("\nConfusion matrix saved.")

if __name__ == "__main__":
    main()
