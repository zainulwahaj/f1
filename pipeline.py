import os
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ─── STATIC PATHS ───────────────────────────────────────────────────────────────
MASTER_CSV   = "f1_master_data.csv"
FEATURES_CSV = "f1_features.csv"
ENCODER_PKL  = "target_encoder.pkl"
MODEL_PKL    = "qualifying_predictor_stack.pkl"

# ────────────────────────────────────────────────────────────────────────────────
def load_master_data(path: str = MASTER_CSV) -> pd.DataFrame:
    """
    Load 'f1_master_data.csv' from disk. Raise FileNotFoundError if not present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master data not found at '{path}'")
    return pd.read_csv(path)


def preprocess_data(master_df: pd.DataFrame, save_to: str = FEATURES_CSV) -> pd.DataFrame:
    """
    From raw 'master_df', produce 'features_df' with all engineered features.
    Optionally save out to FEATURES_CSV for caching.
    """
    df = master_df.copy()

    # 1) Unify constructorId
    df["constructorId"] = (
        df["Constructor.constructorId_race"]
        .fillna(df["Constructor.constructorId_qual"])
    )

    # 2) Parse lap times into seconds
    def parse_time_str(t):
        if pd.isna(t):
            return np.nan
        parts = t.split(":")
        return (int(parts[0]) * 60 + float(parts[1])) if len(parts) == 2 else float(parts)

    for c in ["Q1", "Q2", "Q3", "FastestLap.Time.time"]:
        if c in df.columns:
            df[f"{c}_sec"] = df[c].apply(parse_time_str)

    q_cols = [c for c in df.columns if c.endswith("_sec") and "Q" in c]
    df["bestQualLap_sec"] = df[q_cols].min(axis=1) if q_cols else np.nan

    # 3) Historical driver–circuit aggregates
    hist = (
        df.groupby(["Driver.driverId", "Circuit.circuitId"])
        .agg(
            hist_avg_qpos     = ("position_qual",   "mean"),
            hist_avg_qual_lap = ("bestQualLap_sec", "mean"),
            hist_n_visits     = ("season",          "count"),
            hist_n_dnfs       = ("status",          lambda s: s.str.contains("DNF").sum()),
            hist_q3_rate      = ("Q3_sec",          lambda x: x.notnull().mean())
        )
        .reset_index()
    )
    df = df.merge(hist, on=["Driver.driverId", "Circuit.circuitId"], how="left")

    # 4) Driver–circuit experience depth
    depth = (
        df.groupby(["Driver.driverId", "Circuit.circuitId"])
        .agg(
            hist_visits   = ("round",          "count"),
            hist_podiums  = ("position_qual",  lambda x: (x <= 3).sum()),
            hist_q3_count = ("Q3_sec",         lambda x: x.notnull().sum()),
            hist_avg_laps = ("laps",           "mean")
        )
        .reset_index()
    )
    depth["hist_podium_rate"] = depth["hist_podiums"] / depth["hist_visits"]
    df = df.merge(depth, on=["Driver.driverId", "Circuit.circuitId"], how="left")

    # 5) Rolling driver form (last 3 / last 5)
    df = df.sort_values(["Driver.driverId", "season", "round"])
    df["roll3_avg_qpos"] = (
        df.groupby("Driver.driverId")["position_qual"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["roll3_avg_lap"] = (
        df.groupby("Driver.driverId")["bestQualLap_sec"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["roll5_avg_qpos"] = (
        df.groupby("Driver.driverId")["position_qual"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df["roll3_avg_qpos"].fillna(df["hist_avg_qpos"], inplace=True)
    df["roll3_avg_lap"].fillna(df["hist_avg_qual_lap"], inplace=True)
    df["roll5_avg_qpos"].fillna(df["hist_avg_qpos"], inplace=True)

    df["season_mean_before"] = (
        df.groupby("Driver.driverId")["position_qual"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )
    df["qpos_trend"] = df["position_qual"] - df["season_mean_before"]

    # 6) Constructor rolling form (last 3)
    df = df.sort_values(["constructorId", "season", "round"])
    df["cons_roll3_qpos"] = (
        df.groupby("constructorId")["position_qual"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["cons_roll3_qpos"].fillna(
        df.groupby("constructorId")["position_qual"].transform("mean"), 
        inplace=True
    )

    # 7) Circuit difficulty
    circ = (
        df.groupby("Circuit.circuitId")["position_qual"]
        .mean()
        .reset_index()
        .rename(columns={"position_qual": "circuit_hist_avg_qpos"})
    )
    df = df.merge(circ, on="Circuit.circuitId", how="left")

    # 8) Interaction features
    df["drv_circ_interact"]  = df["hist_avg_qpos"]  * df["roll3_avg_qpos"]
    df["form_cons_interact"] = df["roll3_avg_qpos"] * df["cons_roll3_qpos"]

    # 9) Final feature set & drop‐rows without target
    feature_cols = [
        "hist_avg_qpos", "hist_avg_qual_lap", "hist_n_visits", "hist_n_dnfs", "hist_q3_rate",
        "hist_visits", "hist_podium_rate", "hist_q3_count", "hist_avg_laps",
        "roll3_avg_qpos", "roll3_avg_lap", "roll5_avg_qpos", "qpos_trend",
        "cons_roll3_qpos", "circuit_hist_avg_qpos",
        "drv_circ_interact", "form_cons_interact",
        "Driver.driverId", "Circuit.circuitId", "constructorId"
    ]

    df_features = df.dropna(subset=["position_qual"] + feature_cols)[feature_cols + ["position_qual"]]
    if save_to:
        df_features.to_csv(save_to, index=False)
    return df_features


def train_or_load_model(features_df: pd.DataFrame, model_dir: str = "."):
    """
    Train a stacked ensemble (CatBoost + LightGBM → Ridge) if artifacts are missing;
    otherwise, load existing encoder & model from disk. Return:
      (target_encoder, stacked_model, mae, r2)
    """
    enc_path   = os.path.join(model_dir, ENCODER_PKL)
    model_path = os.path.join(model_dir, MODEL_PKL)

    # If both exist, load them:
    if os.path.exists(enc_path) and os.path.exists(model_path):
        te    = joblib.load(enc_path)
        model = joblib.load(model_path)
        X_enc = te.transform(features_df.drop(columns=["position_qual"]))
        preds = model.predict(X_enc)
        mae   = mean_absolute_error(features_df["position_qual"], preds)
        r2    = r2_score(features_df["position_qual"], preds)
        return te, model, mae, r2

    # Otherwise, train from scratch:
    X = features_df.drop(columns=["position_qual"])
    y = features_df["position_qual"]

    cat_cols = ["Driver.driverId", "Circuit.circuitId", "constructorId"]
    te = TargetEncoder(cols=cat_cols)
    X_enc = te.fit_transform(X, y)
    joblib.dump(te, enc_path)

    cat = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0
    )
    lgb = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        objective="regression_l1",
        n_jobs=-1,
        random_state=42
    )

    stack = StackingRegressor(
        estimators=[("cat", cat), ("lgb", lgb)],
        final_estimator=Ridge(alpha=1.0),
        passthrough=True,
        n_jobs=-1
    )

    stack.fit(X_enc, y)
    joblib.dump(stack, model_path)

    preds = stack.predict(X_enc)
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    return te, stack, mae, r2


def predict_position(
    driver_id: str,
    circuit_id: str,
    constructor_id: str,
    features_df: pd.DataFrame,
    te,
    model
) -> float:
    """
    Given a (driver_id, circuit_id, constructor_id) triple,
    find its row in 'features_df', apply 'te' → 'model', and return a float prediction.
    If no matching row, raise ValueError.
    """
    subset = features_df[
        (features_df["Driver.driverId"]   == driver_id) &
        (features_df["Circuit.circuitId"] == circuit_id) &
        (features_df["constructorId"]     == constructor_id)
    ]
    if subset.empty:
        raise ValueError(f"No data for driver='{driver_id}', circuit='{circuit_id}', constructor='{constructor_id}'")

    X_new     = subset.drop(columns=["position_qual"])
    X_enc_new = te.transform(X_new)
    return float(model.predict(X_enc_new)[0])
