# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 18:16:27 2023

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
   GBDT-1_fast.py uses the Pandas library to feed a large 
   amount of input data into the model for Tm estimation, 
   which is suitable for solving large amounts of input 
   data simultaneously
'''
############################################################
'''
   For the GBDT-1 model, the required input parameters are:
   Year, Doy ,Hour, Lat, Lon, H.
   
   We provide an example input file for testing
'''
############################################################

import joblib
import pandas as pd


#load the GBDT-1 model
model = joblib.load('GBDT_models\gbdt-1.pkl')


#Load input data
data= pd.read_csv("input_example_Scheme1.csv",sep=',')
data_input = data[['Year','Doy','Hour','Lat','Lon','H']].values


#Estimating Tm values using saved model files
y_pred = model.predict(data_input)

# Add the predicted results as a new column and overwrite the entire DataFrame back to the original CSV file
data['Tm_result'] = y_pred
data.to_csv("input_example_Scheme1.csv", index=False)
