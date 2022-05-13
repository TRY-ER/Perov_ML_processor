import pandas as pd
import numpy as np
import joblib 

decoded = pd.read_csv("../outputs/data/smoted_scaled_data.csv")
print(" This is un_decode values below ...")
print(decoded.head(10))

scaler = joblib.load("../inputs/Scalers/standard.z")

un_decoded = scaler.inverse_transform(decoded)
print(" This is un_decode values below ...")
print(un_decoded.head(10))

