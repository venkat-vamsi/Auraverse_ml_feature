import joblib
from sklearn.ensemble import IsolationForest
from utils.scaler import fit_scaler, transform_scaler
from config import STAGE1_FEATURES, CONTAMINATION


def train_stage1(train_df):
    scaler = fit_scaler(train_df, STAGE1_FEATURES)
    train_scaled = transform_scaler(train_df, scaler, STAGE1_FEATURES)

    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=42
    )

    model.fit(train_scaled[STAGE1_FEATURES])

    joblib.dump(model, "stage1_panic_detection/model/model_stage1.pkl")
    joblib.dump(scaler, "stage1_panic_detection/model/scaler_stage1.pkl")

    return model, scaler


def test_stage1(df, model, scaler, save_path=None):
    df_scaled = transform_scaler(df, scaler, STAGE1_FEATURES)

    predictions = model.predict(df_scaled[STAGE1_FEATURES])
    # IsolationForest: -1 = anomaly, 1 = normal

    df_copy = df.copy()
    df_copy["anomaly"] = predictions

    panic_rows = df_copy[df_copy["anomaly"] == -1].drop(columns=["anomaly"])

    if save_path:
        panic_rows.to_csv(save_path, index=False)

    return panic_rows