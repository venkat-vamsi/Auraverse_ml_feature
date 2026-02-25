from sklearn.preprocessing import StandardScaler

def fit_scaler(df, columns):
    scaler = StandardScaler()
    scaler.fit(df[columns])
    return scaler

def transform_scaler(df, scaler, columns):
    df_copy = df.copy()
    df_copy[columns] = scaler.transform(df_copy[columns])
    return df_copy