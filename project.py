import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import os
import random
import time
from sklearn.model_selection import train_test_split 
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras import layers
import tensorflow as tf


def iso_to_hours(duration):
    """
    Convert travel duration column into float.
    
    Args:
    duration: travel duration in ISO format
    Returns:
    Travel duration in hours, float 
    """ 
    # determine if object is ISO format
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration) 
    
    # converts to hours and minutes
    if match:  
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours + minutes / 60
    return None

def red_eye (flight_depart_time): 
    """
    Returns boolean corresponding to whether or not an hours falls between the typical red-eye flight 
    departure hours (between 10PM to 2AM). 

    Args:
    flight_hour: datetime64 object

    Returns: 
    bool: 
    - True, if flight is departs within the time period 
    - False, if flight does not depart within the time period 
    """
    # get hour of departure 
    depart_hour = flight_depart_time.hour
    if (depart_hour >= 22) | (depart_hour <= 2):
        return True 
    return False

def season (flight_date):
    """
    Returns the season a given data falls in. Based on the astronomical 2022 calendar.

    Args:
    flight_date: datetime64 object

    Returns:
    String containing the season (either "Winter", "Spring", "Summer", or "Fall")

    """
    # get month and day of departure 
    month = flight_date.month
    day = flight_date.day

    if (month == 12 & day >= 21) | (month == 1) | (month == 2) | (month == 3 & day < 20):
        return "Winter"

    elif (month == 3) | (month == 4) | (month == 5) | (month == 6 & day < 21):
        return "Spring"

    elif (month == 6) | (month == 7) | (month == 8) | (month == 9 & day < 22):
        return "Summer"

    elif (month == 9) | (month == 10) | (month == 11) | (month == 12):
        return "Fall"

def is_three_days_before_holiday(flight_date, min_date, max_date): 
    """ 
    Returns boolean corresponding wheter or not a flight departs three days or less before a holiday. 
    
    Args:
    flight_date: datetime64 object
    min_date: start of date range
    max_date: end of date range 
    
    Returns: 
    bool: 
    - True, if flight departs five days of less before a holiday
    - False, if flight does not depart five days of less before a holiday
    """ 
    # get 
    from pandas.tseries.holiday import USFederalHolidayCalendar
    us_cal = USFederalHolidayCalendar()
    fed_holidays = us_cal.holidays(min_date, max_date)


    for n in range(1,4): 
        # subtract by n to get date n days before holidays 
        three_days_before_holiday = fed_holidays - pd.Timedelta(days=n)
        # check if date is within range
        if flight_date in three_days_before_holiday:
            return True 
    return False


def red_eye (flight_depart_time): 
    """
    Returns boolean corresponding to whether or not an hours falls between the typical red-eye flight 
    departure hours (between 10PM to 2AM). 

    Args:
    flight_hour: datetime64 object

    Returns: 
    bool: 
    - True, if flight is departs within the time period 
    - False, if flight does not depart within the time period 
    """
    
    # get hour of departure 
    depart_hour = flight_depart_time.hour
    if (depart_hour >= 22) | (depart_hour <= 2):
        return True 
    return False

def season (flight_date):
    """
    Returns the season a given data falls in. Based on the astronomical 2022 calendar.

    Args:
    flight_date: datetime64 object

    Returns:
    String containing the season (either "Winter", "Spring", "Summer", or "Fall")

    """
    # get the month and day of departure
    month = flight_date.month
    day = flight_date.day

    if (month == 12 & day >= 21) | (month == 1) | (month == 2) | (month == 3 & day < 20):
        return "Winter"

    elif (month == 3) | (month == 4) | (month == 5) | (month == 6 & day < 21):
        return "Spring"

    elif (month == 6) | (month == 7) | (month == 8) | (month == 9 & day < 22):
        return "Summer"

    elif (month == 9) | (month == 10) | (month == 11) | (month == 12):
        return "Fall"
    

    
def get_three_days_before_holiday_dates(fed_holidays):
    """ 
    Returns a dict with days within 3 days before a US federal holiday. 
    
    Args: 
    fed_holidays: timeseries with dates of U.S. Fed. Holidays. 
    
    Returns: 
    three_days_before_holiday_dates: dict with all days within 3 days before a US federal holiday. 
    
    Keys: 
    1: Contains set days 1 day before US Fed. Holiday 
    2: Contains set days 2 days before US Fed. Holiday 
    3: Contains set days 3 days before US Fed. Holiday 
    """ 
    
    # create dict 
    three_days_before_holiday_dates = {}

    for n in range(1,4): 
        n_days_before_holiday = fed_holidays - pd.Timedelta(days=n)
        three_days_before_holiday_dates[n] = set(n_days_before_holiday)
        
    return three_days_before_holiday_dates

def remove_outliers (df): 
    """ 
    Removes outliers from dataset using IQR method.
    
    Args: 
    df: dataset to remove outliers from 
    
    Returns: 
    df_clean: df with outliers removed 
    """ 
    
    # Compute quantiles 
    Q1 = df['totalFare'].quantile(0.25)
    Q3 = df['totalFare'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Compute lower and upper bounds 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Compute the number of outliers 
    outliers = df[(df['totalFare'] < lower_bound) | (df['totalFare'] > upper_bound)]
    print(f"\nNumber of outliers detected: {len(outliers)}")
    print(f"Outlier fare threshold: < {lower_bound:.2f} or > {upper_bound:.2f}")

    # Return a cleaned version of the dataset without outliers
    df_clean = df[(df['totalFare'] >= lower_bound) & (df['totalFare'] <= upper_bound)]
    print(f"Number of entries after removing outliers: {len(df_clean)}")
    return df_clean

def train_model(model, X, y, times): 
    """ 
    Splits (80%: 20%), trains, and tests model `n` times. 
    Computes the mean and std MAE.
    
    Args: 
    model: model to train and test
    X: features 
    y: target feature
    times: number of times to train and test the model 
    
    Prints:
    The mean MAE and the std MAE.
    """ 
    mae_scores = np.zeros(times)
        
    for i in range(times): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mae_scores[i] = mae
        
    mean_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    print(f"Mean MAE: {mean_mae}") 
    print(F"STD MAE: {std_mae}") 
    
    

def train_NN_model(model, X_train, y_train, X_test, y_test): 
    """
    Compiles and trains NN model on training samples and evaluates it on test samples.  
    
    Args: 
    model: NN model
    X_train: train features 
    y_train: train target features
    X_test: test features 
    y_test: test target features 
    
    Returns: 
    history: History of the NN model's accuracy & loss on train samples and test samples for each epoch 
    """ 
    
    # compile model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse', 
                  metrics=['mae'])
    
    # fit model
    history = model.fit(X_train, 
                    y_train, 
                    epochs=10, 
                    batch_size = 64,
                    validation_data=(X_test, y_test))
    
    return history 

