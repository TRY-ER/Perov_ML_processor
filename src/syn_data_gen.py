import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTENC

# Loading columns data as form of dictionary
col_data  = joblib.load("../inputs/columns_encoded/syn_data.z")

# Loading dataset 
main_df = pd.read_csv("../inputs/standard_ml_preprocessed_df.csv")

y_str = "PCE_categorical"
y_int = "JV_default_PCE_numeric"

drop_vals = ["JV_average_over_n_number_of_cells_numeric"]


main_df[y_str] = main_df[y_str] * 10 
main_df[y_str] = main_df[y_str].astype(np.int16)
use_df = main_df.copy()
#cleaning the dataframe 
indexes = []
for i,_ in enumerate(use_df.columns):
    index = 0
    for value in use_df.iloc[:,i]:
        if np.isnan(value):indexes.append(index)
        if not np.isfinite(value): indexes.append(index) 
        index += 1
indexes = list(set(indexes))

use_df = use_df.drop(indexes, axis=0)
# for col in use_df.columns:
#     print(col)

# print("\n\n")

# Creating the synthetic data 

# verifying the column values 
# for key,value in col_data.items():
#     if key == "categorical":
#         for val in value:print(val)
#     else:
#         for val in value:print(val)

Y = use_df[y_str]
print(f" shape of Y :  {Y.shape}")
Y_int = use_df[y_int]

use_df = use_df.drop([y_str, y_int], axis =1)
use_df = use_df.drop(drop_vals, axis =1)

X = use_df.copy()
print(f" shape of X :  {X.shape}")

cat_cols = []
for key,value in col_data.items():
    if key == "categorical":
        for val in value:
            if val not in drop_vals:
                cat_cols.append(val)

smote = SMOTENC(categorical_features=cat_cols,
                sampling_strategy="auto",
                random_state=123,
                k_neighbors=3,
                n_jobs=3)

x_smoted, y_smoted = smote.fit_resample(X,Y)
print(main_df.shape)
print(x_smoted.shape)

