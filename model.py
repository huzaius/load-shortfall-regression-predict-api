"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    #df =pd.read_csv('./data/df_train.csv')
    
    
    
    
    #Dropping the features with large amount of outliers
    #df = df.drop(['Barcelona_rain_1h', 
    df = feature_vector_df.drop(['Barcelona_rain_1h',    
    'Seville_rain_1h',           
    'Bilbao_snow_3h',           
    'Barcelona_pressure',      
    'Seville_rain_3h',          
    'Madrid_rain_1h',            
    'Barcelona_rain_3h',      
    'Valencia_snow_3h', 'Bilbao_rain_1h'        
    ], axis = 1)
    
    #Dropping the features with large amount of outliers
    
    
    # create new features by replacing null values
    df_cln_train = df
    df_cln_train['Valencia_pressure'] = df_cln_train['Valencia_pressure'].fillna(df_cln_train['Valencia_pressure'].mode()[0])
    

    #Appropriate datatype for time
    df_cln_train['time'] = pd.to_datetime(df_cln_train['time'])
    

    #Converting Valencia data to numeric
    df_cln_train['Valencia_wind_deg']= df_cln_train['Valencia_wind_deg'].str.extract('(\d+)')
    df_cln_train['Valencia_wind_deg'] = pd.to_numeric(df_cln_train['Valencia_wind_deg'])
    

    #Converting Seville Pressure to numeric
    df_cln_train.Seville_pressure = df_cln_train.Seville_pressure.str.extract('(\d+)')
    
    df_cln_train.Seville_pressure = pd.to_numeric(df_cln_train['Seville_pressure'])
    

    #extracting year month and day for the time column
    df_cln_train.insert(2, 'Year', df_cln_train.time.dt.year) 
    df_cln_train.insert(3,'Month',df_cln_train.time.dt.month)
    df_cln_train.insert(4,'Day',df_cln_train.time.dt.day)

    

    #Dropping Time and Unnamed Columns
    df_cln_train = df_cln_train.drop(['Unnamed: 0', 'time'], axis = 1)
    
    
    df_cln_train = df_cln_train.drop(['Valencia_temp_min', 'Madrid_temp', 'Barcelona_temp', 'Madrid_temp_max' ], axis = 1)
    df_cln_train = df_cln_train.drop(['Bilbao_temp_max', 'Bilbao_temp', 'Madrid_temp_min', 'Seville_temp_min'], axis = 1)
    df_cln_train = df_cln_train.drop(['Valencia_temp', 'Bilbao_temp_min', 'Barcelona_temp_max', 'Seville_temp' ], axis = 1)
    df_cln_train = df_cln_train.drop(['Bilbao_weather_id', 'Valencia_temp_max', 'Seville_temp_max', 'Barcelona_weather_id'], axis = 1)
    df_cln_train = df_cln_train.drop(['Seville_weather_id', 'Madrid_weather_id'], axis = 1)
    df_cln_train = df_cln_train.drop(['Madrid_clouds_all', 'Seville_clouds_all'], axis = 1)
    
    

    feature_vector_df = df_cln_train

    
    predict_vector = feature_vector_df#[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
