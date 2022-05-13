import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler

main_df = pd.read_csv("../inputs/unscaled_preprocessed_df.csv")
col_data = joblib.load("../inputs/columns_encoded/syn_data.z")
cat_cols = col_data["categorical"]
print(main_df.head())

indexes = []
for i,_ in enumerate(main_df.columns):
    index = 0
    for value in main_df.iloc[:,i]:
        if np.isnan(value):indexes.append(index)
        if not np.isfinite(value): indexes.append(index) 
        index += 1
indexes = list(set(indexes))
main_df = main_df.drop(indexes, axis=0)

main_df = main_df.drop(["JV_default_PCE_numeric","JV_average_over_n_number_of_cells_numeric"],axis=1)
Y = main_df["PCE_categorical"]
Y = Y.astype(np.int64)
main_df = main_df.drop(["PCE_categorical"],axis=1)

scl = StandardScaler()
scaled_df = scl.fit_transform(main_df)
scaled_df = pd.DataFrame(scaled_df, columns = main_df.columns)
print(scaled_df.head())

X = scaled_df
cat_col_int = []
index =0
for value in X.columns:
    for cat in cat_cols:
        if cat == value: cat_col_int.append(index)
    index += 1

print(len(cat_col_int))
smote = SMOTENC(categorical_features=cat_col_int,
                sampling_strategy= "minority",
                random_state = 0,
                k_neighbors=3,
                n_jobs=3)

x_smoted, y_smoted = smote.fit_resample(X,Y)
print(x_smoted.shape)
print(y_smoted.shape)

x_smoted_decoded = scl.inverse_transform(x_smoted)
decoded_df = pd.DataFrame(x_smoted_decoded, columns=X.columns)
decoded_df["PCE_categorical"] = y_smoted
print(decoded_df.tail())
decoded_df.to_csv("../inputs/smote_custom_unscaled_processed_df.csv",index=False)

export_df = pd.DataFrame(x_smoted, columns=X.columns)
export_df["PCE_categorical"] = y_smoted
print(export_df.tail())


export_df.to_csv("../inputs/smote_custom_scaled_df.csv",index=False)

# inversing label encoding form deocoded dataframe
main_reversed_df = decoded_df.copy()
for col in decoded_df.columns:
    if col == "PCE_categorical":
        lbl_enc = joblib.load("../inputs/Label_encoders/tar_col.z")
        main_reversed_df[col] = main_reversed_df[col].astype(np.int64)
        main_reversed_df[col] = lbl_enc.inverse_transform(main_reversed_df[col])
    else:
        try:
            lbl_enc = joblib.load(f"../inputs/Label_encoders/{col}.z")
            main_reversed_df[col] = main_reversed_df[col].astype(np.int64)
            main_reversed_df[col] = lbl_enc.inverse_transform(main_reversed_df[col])
        except:pass
main_reversed_df.to_csv("../inputs/smote_main_reversed_df.csv",index=False)
print(main_reversed_df.tail())


