# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 18:14:31 2023

@author: wlgao
"""

############################################################
'''
   The following is the code for using the GBDT-1 model to 
   estimate the weighted mean temperature (Tm). We have 
   used the Joblib library to save the model as a pkl file, 
   and users can install the Joblib library in their own 
   Python environment to use the constructed model
'''
############################################################
'''
   GBDT-1.py defines a function: predict_Tm() to calculate 
   the Tm value. If a large amount of input data is required 
   for calculation, refer to GBDT-1_fast.py
'''
############################################################

import joblib
import pandas as pd
import warnings

# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the GBDT-1 model
model_path = 'GBDT_models\gbdt-1.pkl'
model = joblib.load(model_path)

def predict_tm(year, doy, hour, lat, lon, h):
    """
    Predict Tm value using the provided input features.

    Args:
        year (int): Year.
        doy (int): Day of Year (DOY).
        hour (int): Hour. (0 or 12)
        lat (float): Latitude in degree.
        lon (float): Longitude in degree.
        h (float): H value in m.

    Returns:
        float: Predicted Tm value.
    """
    # Create a DataFrame with the provided input values
    input_data = pd.DataFrame({
        'Year': [year],
        'Doy': [doy],
        'Hour': [hour],
        'Lat': [lat],
        'Lon': [lon],
        'H': [h],
    })

    # Use the model to predict Tm value
    predicted_tm = model.predict(input_data)

    return predicted_tm[0]

def main():
    # Get user inputs
    year = int(input("Enter Year: "))
    doy = int(input("Enter Day of Year (DOY): "))
    hour = int(input("Enter Hour: "))
    lat = float(input("Enter Latitude: "))
    lon = float(input("Enter Longitude: "))
    h = float(input("Enter H (m): "))

    # Predict Tm
    predicted_tm = predict_tm(year, doy, hour, lat, lon, h)
    print("Predicted Tm:", predicted_tm)

if __name__ == "__main__":
    main()