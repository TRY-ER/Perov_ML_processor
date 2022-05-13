import joblib
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder

use_df = pd.read_csv("../inputs/unscaled_preprocessed_df.csv")
print(use_df.head())

scl  = StandardScaler()
new_df = scl.fit_transform(use_df)
new_df = pd.DataFrame(new_df, columns=use_df.columns)
print(new_df.head())

trans_df = scl.inverse_transform(new_df)
trans_df = pd.DataFrame(trans_df, columns=use_df.columns)
print(trans_df.head())


new_df = trans_df.copy()
for col in trans_df.columns:
    if col == "PCE_categorical":
        lbl_enc = joblib.load("../inputs/Label_encoders/tar_col.z")
        new_df[col] = new_df[col].astype(np.int64)
        new_df[col] = lbl_enc.inverse_transform(new_df[col])
    else:
        try:
            lbl_enc = joblib.load(f"../inputs/Label_encoders/{col}.z")
            new_df[col] = new_df[col].astype(np.int64)
            new_df[col] = lbl_enc.inverse_transform(new_df[col])
        except:pass

print(new_df.head())