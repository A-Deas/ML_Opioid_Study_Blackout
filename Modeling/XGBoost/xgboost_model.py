import numpy as np 
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, roc_curve, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import duckdb
import pandas as pd
import json

# --- Preliminaries
DB_PATH = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Data_Tables/DuckDB_Files/ModelReadyBlackout.db"
MODEL_SAVE_DIR = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Saved_Models"
FI_SAVE_DIR = "/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/Feature_Importance/Folded_FI"

RANDOM_SEED = 42
NUM_FOLDS = 12
BETA = 2.0 # if tuning via F-Beta (won't affect anything otherwise)

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

    pd.DataFrame({"Features": feature_cols}).to_csv("/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/XGBoost/feature_cols.csv", index=False)

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

    print(f"Raw  imbalance: {n_pos:,} positives and {n_neg:,} negatives => {n_pos / (n_pos + n_neg):.3%}")
    print(f"Fold imbalance: {n_pos:,} positives and {len(neg_folds[0]):,} negatives => {n_pos / (n_pos + len(neg_folds[0])):.3%}")
    return folds

def find_best_threshold_f1(model, set_data, set_labels):
    probs = model.predict_proba(set_data)[:, 1]

    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0, 1, 101):
        f1 = f1_score(set_labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1 
            best_t = t 
    print(f"\nBest threshold on validation: {best_t:.2f} (F1 = {best_f1:.4f})")
    return best_t

def find_best_threshold_fbeta(model, set_data, set_labels, beta=BETA):
    probs = model.predict_proba(set_data)[:, 1]
    best_t, best_score = 0.5, -1.0

    for t in np.linspace(0, 1, 201):
        preds = (probs >= t).astype(int)
        score = fbeta_score(set_labels, preds, beta=beta, zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t 
    print(f"\nBest threshold on validation: {best_t:.2f} (F-Beta = {best_score:.4f})")
    return best_t

def find_best_threshold_youden(model, set_data, set_labels):
    probs = model.predict_proba(set_data)[:, 1]

    fpr, tpr, thresholds = roc_curve(set_labels, probs)

    # Youden Index = sensitivity + specificity - 1
    # sensitivity = tpr, specificity = (1 - fpr)
    youden = tpr + (1 - fpr) - 1

    best_thresh = thresholds[np.argmax(youden)]

    print(f"\nYouden Index threshold: {best_thresh:.3f}, "
          f"(sensitivity = {tpr[np.argmax(youden)]:.3f}, "
          f"specificity = {(1-fpr[np.argmax(youden)]):.3f})")
    return float(best_thresh)

def evaluate(model, fold_idx, set_data, set_labels, set_name, thresh):
    probs = model.predict_proba(set_data)[:, 1]
    preds = (probs >= thresh).astype(int)

    acc = accuracy_score(set_labels, preds)
    bal_acc = balanced_accuracy_score(set_labels, preds)
    prec = precision_score(set_labels, preds)
    recall = recall_score(set_labels, preds)
    f1 = f1_score(set_labels, preds, zero_division=0)
    fbeta = fbeta_score(set_labels, preds, beta=BETA, zero_division=0)
    auc_roc = roc_auc_score(set_labels, probs)
    auc_pr = average_precision_score(set_labels, probs)

    tn, fp, fn, tp = confusion_matrix(set_labels, preds).ravel()

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    nne = 1 / ppv

    metrics = {
        "Fold": fold_idx,
        "Accuracy": acc,
        "Balanced accuracy":bal_acc,
        "Precision": prec,
        "Recall": recall,
        "F1": f1,
        "F-Beta": fbeta,
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        # "PPV": ppv,
        # "NPV": npv,
        # "NNE": nne,
    }

    print(f"[{set_name}] | Acc = {acc:.4f} | Bal acc = {bal_acc:.4f} | "
          f"Prec = {prec:.4f} | Recall = {recall:.4f} | F1 = {f1:.4f} | "
          f"F-Beta = {fbeta:.4f} | AUC ROC = {auc_roc:.4f} | AUC PR = {auc_pr:.4f}")
        #   f"PPV = {ppv:.4f} | NPV = {npv:.4f} | NNE = {nne:.4f}")
    return metrics

def main():
    df = load_data()
    X, y, feature_cols = prepare_features(df)

    sid_to_idx = {sid: i for i, sid in enumerate(df["PatientSID"].values)}

    folds = build_folds(df)

    results = []

    for fold_idx, fold_sids in enumerate(folds, start=1):
        print(f"\n----------------- Fold {fold_idx}/{NUM_FOLDS} -----------------")
        
        idxs = [sid_to_idx[sid] for sid in fold_sids]
        fold_X = X.iloc[idxs].reset_index(drop=True)
        fold_y = y[idxs]

        # scale_pos_weight = (fold_y == 0).sum() / (fold_y == 1).sum()

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

        print(f"\nData shapes: train = {train_X.shape} | validation = {val_X.shape} | test = {test_X.shape}")

        # Train the XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective = "binary:logistic",
            eval_metric = "auc", # for binary probabilistic modeling

            max_depth = 6,
            learning_rate = 0.027936163058363116,
            n_estimators = 500,
            min_child_weight = 14,
            subsample = 0.5704133977342001,
            colsample_bytree = 0.5235739953589245,
            gamma = 1.2783409290641894,
            reg_lambda = 1.9841099665648618,
            reg_alpha = 0.5126722880820559,
            scale_pos_weight = 100,
            max_delta_step = 1,

            n_jobs = -1, # use all CPU cores
            random_state = RANDOM_SEED,
            tree_method = "hist",
            enable_categorical = True,
            early_stopping_rounds = 20,
        )

        xgb_model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
            verbose=False,
        )

        # Save the xgb model for each fold
        xgb_model.save_model(f"{MODEL_SAVE_DIR}/trained_xgb_fold_{fold_idx}.json")

        # Evaluate without threshold testing first
        print("\nEvaluation without threshold testing: ")
        _ = evaluate(xgb_model, fold_idx, set_data=test_X, set_labels=test_y, set_name="Test", thresh=0.5)

        # best_thresh = find_best_threshold_f1(xgb_model, set_data=val_X, set_labels=val_y)
        # metrics = evaluate(xgb_model, fold_idx, set_data=test_X, set_labels=test_y, set_name="Test (threshold tuned via F1)", thresh=best_thresh)

        best_thresh = find_best_threshold_fbeta(xgb_model, set_data=val_X, set_labels=val_y)
        metrics = evaluate(xgb_model, fold_idx, set_data=test_X, set_labels=test_y, set_name="Test (threshold tuned via F-Beta)", thresh=best_thresh)
        results.append(metrics)

        # Feature Importance per fold
        importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        importances = importances.round(4)
        importances.to_csv(f"{FI_SAVE_DIR}/xgb_FI_fold_{fold_idx}.csv")
        print("\nFeature Importances:")
        print(importances.head(30))

    df = pd.DataFrame(results)
    df.to_csv(f"/mnt/pfb_edhdata/pfb_edhdata_bps/deasa/ML_Opioid_Study_FB/Modeling/XGBoost/tuned_xgb_fold_metrics.csv", index=False)
    print(f"\n----------------- Average performance across all {NUM_FOLDS} folds -----------------")
    print(df.mean(numeric_only=True))

    # Compute feature importance across all folds
    all_importances = []

    for fold_idx in range(1, NUM_FOLDS + 1):
        fi_df = pd.read_csv(f"{FI_SAVE_DIR}/xgb_FI_fold_{fold_idx}.csv", index_col=0)
        fi_df.columns = [f"FI_fold_{fold_idx}"]
        all_importances.append(fi_df)

    fi_merged = pd.concat(all_importances, axis=1)

    fi_merged["mean_importance"] = fi_merged.mean(axis=1)
    fi_merged = fi_merged.sort_values("mean_importance", ascending=False)
    fi_merged = fi_merged.round(4)

    print(f"\n---------- Average feature importance across all {NUM_FOLDS} folds ----------")
    print(fi_merged["mean_importance"].head(30))
    fi_merged.to_csv(f"{FI_SAVE_DIR}/xgb_FI_averaged.csv", index_label="feature")

if __name__ == "__main__":
    main()