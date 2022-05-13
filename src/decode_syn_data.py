import pandas as pd
import numpy as np
import joblib 

decoded = pd.read_csv("../outputs/data/smoted_scaled_data.csv")
print(" This is un_decode values below ...")
print(decoded.head(10))

scaler = joblib.load("../inputs/Scalers/standard.z")

un_decoded = scaler.inverse_transform(decoded)
print(" This is un_decode values below ...")
un_decoded = pd.DataFrame(un_decoded, columns=decoded.columns)
print(un_decoded.head(10))


renew_df = pd.DataFrame()
# for col in un_decoded.columns:
#     try:
#         lbl_enc = joblib.load(f"../inputs/Label_encoders/{col}.z")
#         renew_df[col] = lbl_enc.inverse_transform(un_decoded[col])
#     except:
#         renew_df[col] = un_decoded[col]
# print("renew_values are below")
# print(renew_df.tail())

lbl_enc = joblib.load("../inputs/Label_encoders/Substrate_stack_sequence_1.z")
renew_df["Substrate_stack_sequence_1"] = lbl_enc.inverse_transform(un_decoded["Substrate_stack_sequence_1"])
print(renew_df.head())