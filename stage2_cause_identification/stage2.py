import joblib
from sklearn.cluster import KMeans
from utils.scaler import fit_scaler, transform_scaler
from config import STAGE2_FEATURES, STAGE2_K


def train_stage2(train_panic_df):
    scaler = fit_scaler(train_panic_df, STAGE2_FEATURES)
    train_scaled = transform_scaler(train_panic_df, scaler, STAGE2_FEATURES)

    model = KMeans(n_clusters=STAGE2_K, random_state=42)
    model.fit(train_scaled[STAGE2_FEATURES])

    joblib.dump(model, "stage2_cause_identification/model/model_stage2.pkl")
    joblib.dump(scaler, "stage2_cause_identification/model/scaler_stage2.pkl")

    return model, scaler


def test_stage2(test_panic_df, model, scaler, save_path=None):
    df_scaled = transform_scaler(test_panic_df, scaler, STAGE2_FEATURES)
    clusters = model.predict(df_scaled[STAGE2_FEATURES])

    df = test_panic_df.copy()
    df["cause_cluster"] = clusters

    centers = model.cluster_centers_
    cluster_names = {}

    for i, center in enumerate(centers):
        lux, sound, humidity = center
        if lux > sound and lux > humidity:
            cluster_names[i] = "Light-triggered"
        elif sound > lux and sound > humidity:
            cluster_names[i] = "Sound-triggered"
        else:
            cluster_names[i] = "Environmental-triggered"

    df["cause_name"] = df["cause_cluster"].map(cluster_names)

    if save_path:
        df.to_csv(save_path, index=False)

    return df