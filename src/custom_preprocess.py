import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


main_df = pd.read_csv("../inputs/smote_main_reversed_revised.csv")
col_data = joblib.load("../inputs/columns_encoded/syn_data.z")


cat_cols = col_data["categorical"]
tar_col = "PCE_categorical"


# using label encoding
trans_df = main_df.copy()
for col in main_df.columns:
    if col in cat_cols:
        print(col)
        lbl_enc = LabelEncoder()
        trans_df[col] = lbl_enc.fit_transform(trans_df[col])
        joblib.dump(lbl_enc,f"../outputs/smote_label_enc/{col}.z")
    elif col == tar_col:
        lbl_enc = LabelEncoder()
        trans_df[col] = lbl_enc.fit_transform(trans_df[col])
        joblib.dump(lbl_enc,f"../outputs/smote_label_enc/{col}.z")

trans_df.to_csv("../outputs/data/trainable_encoded.csv",index=False)

Y = trans_df[tar_col]
X = trans_df.drop([tar_col],axis=1)
scl = StandardScaler()
X_trans = scl.fit_transform(X)

trainable_df = pd.DataFrame(X_trans, columns=X.columns)
trainable_df[tar_col] = Y

trainable_df.to_csv("../outputs/data/trainable_scaled.csv",index=False)
