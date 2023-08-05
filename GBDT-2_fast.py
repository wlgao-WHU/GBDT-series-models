# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 18:11:12 2023

@author: wlgao
"""

############################################################
'''
   The following is the code for using the GBDT-2 model to 
   estimate the weighted mean temperature (Tm). We have 
   used the Joblib library to save the model as a pkl file, 
   and users can install the Joblib library in their own 
   Python environment to use the constructed model
'''
############################################################
'''
   GBDT-2_fast.py uses the Pandas library to feed a large 
   amount of input data into the model for Tm estimation, 
   which is suitable for solving large amounts of input 
   data simultaneously
'''
############################################################
'''
   For the GBDT-2 model, the required input parameters are:
   Year, Doy ,Hour, Lat, Lon, H, Temp.
   
   We provide an example input file for testing
'''
############################################################

import joblib
import pandas as pd


#load the GBDT-2 model
model = joblib.load('GBDT_models\gbdt-2.pkl')


#Load input data
data= pd.read_csv("input_example_Scheme2.csv",sep=',')
data_input = data[['Year','Doy','Hour','Lat','Lon','H','Temp']].values


#Estimating Tm values using saved model files
y_pred = model.predict(data_input)

# Add the predicted results as a new column and overwrite the entire DataFrame back to the original CSV file
data['Tm_result'] = y_pred
data.to_csv("input_example_Scheme2.csv", index=False)
