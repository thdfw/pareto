import pandas as pd
import numpy as np
import os
import sys
import csv
from forecaster import fcLib
from forecaster import fcSelector
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

PLOT = False
PRINT = False
library = fcLib.forecasters(fcLib.forecaster_list)

# Create a CSV file to store past results
if not os.path.isfile(os.getcwd()+'/data/best_forecasters.csv'):
    with open(os.getcwd()+'/data/best_forecasters.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Data file', 'Best forecaster'])
        
# ---------------------------------------------------------------------
# Import data
# ---------------------------------------------------------------------

def get_past_data(path):

    # Import yearly load and outside temperature data from GridWorks
    df = pd.read_excel(path, header=3, index_col = 0)
    df.index = pd.to_datetime(df.index)
    df.index.name = None

    # Rename columns
    renamed_columns = {
        'Outside Temp F': 'T_OA',
        'House Power Required AvgKw': 'Q_load'}
    df.rename(columns=renamed_columns, inplace=True)

    # Convert outside air temperature from °F to °C
    df['T_OA'] = df['T_OA'].apply(lambda x: round(5/9 * (x-32),2))

    # Keep only date, weather, and load
    df = df[['T_OA','Q_load']]#[:1000]

    if PRINT: print(f"\nSuccesfully read past hourly weather and load data ({len(df)} hours)")
    
    return df

# ---------------------------------------------------------------------
# Try different models to forecast the load based on weather
# ---------------------------------------------------------------------

def get_best_forecaster(df, path_to_past_data):

    # **** To obtain the best forecaster via fcSelector ****
    fcselec = False
    if fcselec:
    
        best_forecaster = FcSelect(path_to_past_data)
        library = fcLib.forecasters(fcLib.forecaster_list)
        
        for forecaster in library.forecasters:
            if forecaster['name'] != best_forecaster: continue
            
            X = df[['T_OA']]
            y = df[['Q_load']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 42)
            
            model = getattr(fcLib, forecaster['fun'])(**forecaster['parameter'])
            model.fit(X_train, y_train)
        
        return best_forecaster, model
    # **** End obtain the best forecaster based on fcSelector ****

    # Check if the best forecaster has already been found before
    path_to_past_data = path_to_past_data.split('/')[-1]
    if PRINT: print(f"Past data file: {path_to_past_data}")
    
    # Read best forecast from CSV
    best_forecaster_csv = ""
    if path_to_past_data != "":
        with open(os.getcwd()+'/data/best_forecasters.csv', 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if row and row[0] == path_to_past_data:
                    best_forecaster_csv = row[1]
    
    if PRINT:
        if best_forecaster_csv != "":
            print(f"Best forecaster is known for this past data: {best_forecaster_csv}.")
        else:
            print(f"The best forecaster is not yet known for this past data.")

    # Split the data into X (weather) and y (load)
    X = df[['T_OA']]
    y = df[['Q_load']]

    # Create training and testing data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 42)

    # Create a dict to store prediction values for a 48-hour plot
    scores = {}

    # Iterate through each of the forecaster models
    if PRINT: print("\nTrying different models...")
    library = fcLib.forecasters(fcLib.forecaster_list)
    skipped_forecasters = ['todt', 'sarimax_with_forecast']

    for forecaster in library.forecasters:
    
        # If you already know the best forecaster, skip all others
        if best_forecaster_csv != "" and forecaster['name'] != best_forecaster_csv: continue

        # Skip forecasters that bug
        if PRINT: print(f"- {forecaster['name']}")
        if forecaster['name'] in skipped_forecasters:
            if PRINT: print("--- skipped ---")
            continue

        # Fit the model to the training data and predict for the testing data
        model = getattr(fcLib, forecaster['fun'])(**forecaster['parameter'])
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        
        # Calculate the rmse and store it in the associated dataframe
        rmse = np.sqrt(mean_squared_error(y_test, predict))
        scores[forecaster['name']] = rmse
        
    # Find the best forecaster
    rmse_list = [value for key, value in scores.items()]
    name_list = [key for key, value in scores.items()]
    best_forecaster = name_list[rmse_list.index(min(rmse_list))]
    if PRINT: print(f"\nThe forecaster that performed best on the past data is {best_forecaster}.")
    
    # Save the result in a csv file for future reference
    with open(os.getcwd()+'/data/best_forecasters.csv', 'r+', newline='') as csvfile:
        
        # Check if a best forecaster has already been found for this data
        already_saved = False
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[:2] == [path_to_past_data, best_forecaster]:
                already_saved = True
                
        # If not, save the new result
        if not already_saved:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([path_to_past_data, best_forecaster])
            if PRINT: print("Saved the best forecaster in the CSV file.")
    
    # Save the model for the best forecaster
    for forecaster in library.forecasters:
        if forecaster['name'] != best_forecaster: continue
        model = getattr(fcLib, forecaster['fun'])(**forecaster['parameter'])
        model.fit(X_train, y_train)
        
    # Plot the performance of the forecaster on the first 48 hours
    if PLOT:
    
        data_plot = []

        for i in range(48):
            
            forecast = [[X.T_OA[i].tolist()]]
            predict = model.predict(forecast)
            predict = predict[0] if best_forecaster == 'gradient_boosting' else predict

            data_plot.append(predict)

        plt.figure(figsize=(15,5))
        plt.plot(df.Q_load[0:48].tolist(), label="reality")
        plt.plot(data_plot, label=f"{best_forecaster} prediction", alpha=0.6, linestyle='dashed')
        plt.xlabel("Time")
        plt.ylabel("Load")
        plt.legend()
        plt.show()

    return best_forecaster, model

# ---------------------------------------------------------------------
# Get (1-delta)*100 CI using split conformal prediction
# ---------------------------------------------------------------------

def get_confidence_interval(best_forecaster, delta, path):

    # If we already know the confidence interval for this past data
    path_to_past_data = path.split('/')[-1]
    with open(os.getcwd()+'/data/best_forecasters.csv', 'r+', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[:3] == [path_to_past_data, best_forecaster, str(delta)] and row[3]!="":
                if PRINT: print(f"The CI width has already been found to be {row[3]}")
                return(float(row[3]))

    df = get_past_data(path)
    X = df[['T_OA']]
    y = df[['Q_load']]

    # Split the data into training and holdout sets of same size n (8760/2 = 4380)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, train_size = 0.5, random_state = 42)
    n = len(X_holdout)
    if PRINT: print(f"\nThe size of the holdout and training sets is n = {len(X_holdout)}")
    
    # Train the model with the training set
    for forecaster in library.forecasters:

        # Use the requested forecaster
        if forecaster['name'] != best_forecaster: continue

        # Fit the model to the training data
        model = getattr(fcLib, forecaster['fun'])(**forecaster['parameter'])
        model.fit(X_train, y_train)
        
        # Use the holdout data to test the model and get residuals R_1,..., R_n
        trues, predicts, residuals = [], [], []
        percentage_list = [25, 50, 75]
        if PRINT: print("Computing residuals...")
        for i in range(len(X_holdout)):
            predict = model.predict([[X_holdout.T_OA[i].tolist()]])[0]
            true = y_holdout.Q_load.iloc[i]
            residual = np.abs(true-predict)
            trues.append(true)
            predicts.append(predict)
            residuals.append(residual)
            if int(i/n*100) in percentage_list:
                if PRINT: print(f"... {round(i/n*100)}%")
                percentage_list = percentage_list[1:] if len(percentage_list)>1 else []
        if PRINT: print("Done.\n")

        # Sort the residuals
        residuals.sort(reverse=False)
        
        # Get the confidence interval for a new point
        new_point = [[5]]
        predict = model.predict(new_point)[0]
        CI_width = residuals[int((1-delta)*(n+1))-1] * 2
        CI_min = predict - CI_width/2
        CI_max = predict + CI_width/2
        if PRINT: print(f"The width of the {(1-delta)*100}% confidence interval is {round(CI_width,4)}")

    # Plot the confidence interval over 50 hours
    if PLOT:
                
        plt.figure(figsize=(14,5))
        plot_length = 50

        predicted = []
        upper_bounds = []
        lower_bounds = []
        errors_index = [np.nan]*plot_length
        errors_count = 0
        
        for i in range(plot_length):
            
            # Get the forecast and the prediction
            forecast = [[df.T_OA[i].tolist()]]
            predict = model.predict(forecast)[0]
            
            # Get the confidence interval by conformal prediction
            CI_min = predict - residuals[int((1-delta)*(n+1))-1]
            CI_max = predict + residuals[int((1-delta)*(n+1))-1]
            
            predicted.append(predict)
            lower_bounds.append(CI_min)
            upper_bounds.append(CI_max)

            # If the true value is outside the confidence interval count error
            if df.Q_load.iloc[i] > CI_max or df.Q_load.iloc[i] < CI_min:
                errors_index[i] = df.Q_load.iloc[i]
                errors_count += 1

        # Plot
        plt.plot(df.Q_load[0:plot_length].tolist(), color='blue', alpha=0.7, label="Real load", linestyle='dashed')
        plt.plot(predicted, color='black', alpha=0.4, label='Predicted load')
        plt.fill_between(range(plot_length), lower_bounds, upper_bounds, color='gray', alpha=0.1, label=f'{round((1-delta)*100,1)}% confidence interval')
        plt.scatter(range(plot_length), errors_index, marker='o', color='red', label='Real load outside of CI')
        plt.xlabel("Time [hours]")
        plt.ylabel("Load [kWh]")
        plt.legend()
        plt.show()
        
        print(f"{errors_count} loads ({round(errors_count/plot_length*100,2)}% of {plot_length}) are outside of the predicted {round((1-delta)*100,1)}% confidence interval")
        
    # Save the result in a csv file for future reference
    with open(os.getcwd()+'/data/best_forecasters.csv', 'r+', newline='') as csvfile:
        
        # Check if a best forecaster has already been found for this data
        already_saved = False
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[:3] == [path_to_past_data, best_forecaster, delta]:
                already_saved = True
                
        # If not, save the new result
        if not already_saved:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([path_to_past_data, best_forecaster, delta, CI_width])
            if PRINT: print("Saved the confidence interval width in the CSV file.")
        
    return CI_width
    
# ---------------------------------------------------------------------
# Get the forecast with confidence interval
# ---------------------------------------------------------------------

def get_forecast_CI(weather, best_forecaster, model, delta, path):

    CI_width = get_confidence_interval(best_forecaster, delta, path)
    
    predictions, CI_min_load, CI_max_load = [], [], []
    
    for i in range(len(weather)):
        
        predict = model.predict([[weather[i]]])
        
        predictions.append(predict)
        CI_min_load.append(predict - CI_width/2)
        CI_max_load.append(predict + CI_width/2)
        
    if PRINT: print(f"\nFinished predicting the load with {best_forecaster}.")
    if PLOT:
                    
        # Plot the weather with the confidence interval
        preds = [round(x[0],2) for x in predictions]
        CI_min = [round(x[0],2) for x in CI_min_load]
        CI_max = [round(x[0],2) for x in CI_max_load]
        plt.plot(preds, color='red', alpha=0.6, label='Load forecast')
        plt.fill_between(range(len(preds)), CI_min, CI_max, color='red', alpha=0.1, label='90% confidence interval')
        plt.xlabel("Hour")
        plt.ylabel("Load (kWh)")
        plt.xticks(list(range(len(predictions))))
        plt.legend()
        plt.show()

    return predictions, CI_min_load, CI_max_load


def FcSelect(path):

    library = fcLib.forecasters(fcLib.forecaster_list)
    fcList = fcLib.forecaster_list

    # Import yearly load and outside temperature data from GridWorks
    df = pd.read_excel(path, header=3, index_col = 0)
    df.index = pd.to_datetime(df.index)
    df.index.name = None
    if PRINT: print("Sucesfully read the data file.")

    # Rename columns
    renamed_columns = {
        'Outside Temp F': 'weather-oat',
        'House Power Required AvgKw': 'Q_load'}
    df.rename(columns=renamed_columns, inplace=True)

    # Convert outside air temperature from °F to °C
    df['weather-oat'] = df['weather-oat'].apply(lambda x: round(5/9 * (x-32),2))
    df['dataValid'] = True

    # Keep only date, weather, and load
    data = df[['weather-oat', 'dataValid', 'Q_load']]

    # Split the data into X and y
    X_columns = [col for col in data.columns if not 'load' in col]
    y_columns = 'Q_load'

    # package data for framework
    data_eval = {
        'X': data[X_columns],
        'y': data[y_columns]
    }

    default_params = {'train_size': 0.75, 'train_method': 'train_test_split'}
    params = default_params.copy()

    a = fcSelector.ForecasterFramework(params=params, data=data_eval, fcList=fcList)

    a.evaluateAll(parallel=False)
    
    if PRINT:
        print(f'best forecaster: {a.bestModelName}')
        print(f'score: {a.bestScore}')
        print(a.fcData.sort_values('score', ascending=False))

    return a.bestModelName
