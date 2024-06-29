import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime as dtm
import math
import load_forecast, pareto_algorithm, fmu_simulation, weather_forecast

# Rendering
PLOT = True
PRINT = True

# All lists for final plot
Q_HP_list = []
Q_HP_expected_list = []
load_list = []
SOC_list = []
c_el_list = []

# Get forecast for Maine (= None for live forecast)
start, end = None, None
start = dtm.datetime(2024, 5, 24, 0, 0, 0)
end = start + dtm.timedelta(hours=50)
weather_total, CI_weather = weather_forecast.get_weather(start, end)

# Treat NaNs
for i in range(len(weather_total)):
    if math.isnan(weather_total[i]):
        print(f"\nWARNING: A NaN was found at hour {i} of the weather forecast provided by pvlib.")
        if i>0:
            weather_total[i] = weather_total[i-1]
        else:
            weather_total[i] = 100

# ---------------------------------------
# Parameters
# ---------------------------------------

# Temperature of water going to the HPs (Celcius)
T_HP_in = 57

# Minimum temperature of water going to the PCM (Celcius)
T_sup_HP_min = 58

# Maximum storage capacity (kWh)
max_storage = 20

# Select price forecast (options are: "GridWorks", "Progressive", "Peter")
price_forecast = "CFH"

# ---------------------------------------
print("0 - Find the best forecaster")
# ---------------------------------------

# Get past data
path_to_past_data = os.getcwd()+'/data/gridworks_yearly_data.xlsx'
past_data = load_forecast.get_past_data(path_to_past_data)

# Find the forecaster that performs best on past data
best_forecaster, model = load_forecast.get_best_forecaster(past_data, path_to_past_data)

print(f"The forecaster that performed best on the past data is {best_forecaster}.")

print("\n---------------------------------------")
print("-- RUN PARETO MPC --")
print("---------------------------------------")

# Initial state
SoC_0 = 0

# Number of MPC simulation hours
num_iterations = 24

for iter in range(num_iterations):
    
    print("\n---------------------------------------")
    print(f"Hour {iter}:00-{iter+1}:00")
    print("---------------------------------------")
    
    print(f"Current storage level: {SoC_0} kWh")

    # ---------------------------------------
    print("1 - Get weather forecast")
    # ---------------------------------------

    '''
    # Get forecast for Maine (= None for live forecast)
    start, end = None, None
    start = dtm.datetime(2024, 3, 6, 0, 0, 0) + dtm.timedelta(hours=iter)
    end = start + dtm.timedelta(hours=17)
    weather, CI_weather = weather_forecast.get_weather(start, end)

    # Treat NaNs
    for i in range(len(weather)):
        if math.isnan(weather[i]):
            print(f"\nWARNING: A NaN was found at hour {i} of the weather forecast provided by pvlib.")
            if i>0:
                weather[i] = weather[i-1]
            else:
                weather[i] = 100
    '''
    
    weather = weather_total[iter:iter+17]
    print([round(x,1) for x in weather])

    # Lenght of simulation (hours)
    num_hours = len(weather)
    CI_weather = CI_weather[:num_hours]

    if PRINT: print(f"\n{num_hours}-hour weather forecast succesfully obtained, with 95% confidence interval.")
    if PRINT: print(f"{weather} \n+/- {CI_weather[:num_hours]}°C")
    
    # ---------------------------------------
    print("2 - Get forecasted load")
    # ---------------------------------------

    # Predict load with 95% confidence interval for predicted weather
    delta = 0.05
    pred_load, min_pred_load, max_pred_load = load_forecast.get_forecast_CI(weather, best_forecaster, model, delta, path_to_past_data)
    CI_load = pred_load[0]-min_pred_load[0]
    if PRINT: print(f"\nLoad succesfully predicted with {best_forecaster}, with {round((1-delta)*100)}% confidence interval.")
    if PRINT: print(f"{[round(x[0],2) for x in pred_load]} \n+/- {CI_load} kWh")
    
    #pred_load = [0.8*x for x in pred_load]

    # Predict load with 95% confidence interval for coldest predicted weather (weather - CI)
    min_weather = [round(weather[i]-CI_weather[i],2) for i in range(num_hours)]
    pred_max_load, min_pred_max_load, max_pred_max_load = load_forecast.get_forecast_CI(min_weather, best_forecaster, model, delta, path_to_past_data)

    # The final confidence interval for the load is CI(forecast->weather) + CI(weater->load)
    final_CI = [max_pred_max_load[i]-pred_load[i] if CI_weather[i]>0 else [0] for i in range(num_hours)]
    final_CI = [round(x[0],2) for x in final_CI]
    if PRINT: print(f"\nCombining with weather confidence interval:")
    if PRINT: print(f"{[round(x[0],2) for x in pred_load]} \n+/- {final_CI} kWh")

    # PLOT
    if PLOT:
        fig, ax = plt.subplots(2,1, figsize=(8,5), sharex=True)

        ax[0].set_xlabel("Time [hours]")
        ax[0].set_ylabel("Load [kWh]")

        pred_min_load = [pred_load[i][0] - (pred_max_load[i][0] - pred_load[i][0]) for i in range(len(pred_load))]
        pred_max_load = [x[0] for x in pred_max_load]

        ax[0].plot(pred_load, color='red', alpha=0.8, label='Load')
        ax[0].fill_between(range(len(pred_load)), pred_min_load, pred_max_load, color='red', alpha=0.1, label='Weather CI')

        min1 = [pred_max_load[i]-CI_load[0] for i in range(len(pred_load))]
        max1 = [pred_max_load[i]+CI_load[0] for i in range(len(pred_load))]

        min2 = [pred_min_load[i]-CI_load[0] for i in range(len(pred_load))]
        max2 = [pred_min_load[i]+CI_load[0] for i in range(len(pred_load))]

        ax[0].fill_between(range(len(pred_load)), min1, max1, color='blue', alpha=0.1, label='Forecaster CI on max/min weather')
        ax[0].fill_between(range(len(pred_load)), min2, max2, color='blue', alpha=0.1)

        ax[0].plot([pred_max_load[i]+CI_load for i in range(len(pred_load))], color='red', alpha=0.8, linestyle='dotted', label='Overall CI')
        ax[0].plot([pred_min_load[i]-CI_load for i in range(len(pred_load))], color='red', alpha=0.8, linestyle='dotted')
        ax[0].set_ylim([max([pred_max_load[i]+CI_load for i in range(len(pred_load))])+2.5, min([pred_min_load[i]-CI_load for i in range(len(pred_load))])-2.5])
        ax[0].legend()

        ax[1].set_xlabel("Time [hours]")
        ax[1].set_ylabel("Outside air temperature [°C]")

        ax[1].plot(weather, color='gray', alpha=0.8, label='Outside air temperature [°C]')

        min_weather = [weather[i]-CI_weather[i] for i in range(len(CI_weather))]
        max_weather = [weather[i]+CI_weather[i] for i in range(len(CI_weather))]
        ax[1].fill_between(range(len(pred_load)), min_weather, max_weather, color='gray', alpha=0.1, label='Weather CI')
        ax[1].set_ylim([max(max_weather)+10,min(min_weather)-10])
        ax[1].legend()

        plt.show()
    
    # ---------------------------------------
    print("3 - Get HP commands from Pareto")
    # ---------------------------------------

    # Get the operation over the forecsats
    Q_HP, m_HP = pareto_algorithm.get_pareto(pred_load, price_forecast, weather, T_HP_in, T_sup_HP_min, max_storage, False, False, num_hours, final_CI, iter, SoC_0)

    if PRINT: print(f"Obtained the solution from the pareto algorithm.\nQ_HP = {Q_HP}")

    # Is the heat pump on or off
    delta_HP = [0 if q==0 else 1 for q in Q_HP]
    if PRINT: print(f"\nConverted Q_HP into commands for the FMU:\ndelta_HP = {delta_HP}")

    # What temperature setpoint should we give it
    T_sup_HP = [round(Q_HP[i]*1000/m_HP[i]/4187 + T_HP_in,1) if Q_HP[i]!=0 else np.nan for i in range(len(Q_HP))]
    if PRINT: print(f"T_sup_HP = {T_sup_HP}")
    
    # To avoid sending NaNs to the FMU we set the setpoint temperature at 40°C when the HP is off
    T_sup_HP = [round(Q_HP[i]*1000/m_HP[i]/4187 + T_HP_in,1) if Q_HP[i]!=0 else 40 for i in range(len(Q_HP))]
        
    # ---------------------------------------
    print("4 - Send commands to FMU and simulate")
    # ---------------------------------------

    num_hours = 1

    # Send commands and obtain simulation results
    df = fmu_simulation.simulate(delta_HP, T_sup_HP, weather, num_hours, pred_load, iter)
    df = df.drop(df.index[-1])

    # Convert SOC to kWh
    df['SOC'] = df['SOC'] * max_storage
    df['SOC'] = df['SOC'].round(2)
    
    # Compute real Q_HP
    df['m_HP'] = (df['T_HP_sup']-273).apply(fmu_simulation.m_HP)
    df['Q_HP'] = df['HeatPumpOnOff'] * df['m_HP'] * 4187 * (df['T_HP_sup'] - df['T_HP_ret'])
    df['Q_HP'] = df['Q_HP'] / 1000
    df['Q_HP'] = df['Q_HP'].round(2)
    df['HeatPumpOnOff'] = df['HeatPumpOnOff'].round()
    df['T_HP_ret'] = df['T_HP_ret'].round(1)
    df['T_HP_sup'] = df['T_HP_sup'].round(1)
    df.drop('m_HP', axis=1, inplace=True)

    # Add inputs to df
    df['INPUT_delta_HP'] = [x for x in delta_HP[:num_hours] for _ in range(60)]
    df['INPUT_T_HP_sup_setpoint'] = [x+273 for x in T_sup_HP[:num_hours] for _ in range(60)]

    # Compute expected Q_HP
    df['m_HP_expected'] = (df['INPUT_T_HP_sup_setpoint']-273).apply(fmu_simulation.m_HP)
    df['Q_HP_expected'] = df['INPUT_delta_HP'] * df['m_HP_expected'] * 4187 * (df['INPUT_T_HP_sup_setpoint'] - T_HP_in - 273)
    df['Q_HP_expected'] = df['Q_HP_expected'] / 1000
    df['Q_HP_expected'] = df['Q_HP_expected'].round(2)
    df.drop('m_HP_expected', axis=1, inplace=True)

    if PRINT: print(df)

    # Electricity prices and load
    # c_el_crop = [x for x in c_el[:num_hours] for _ in range(60)]
    load = [x for x in pred_load[:num_hours] for _ in range(60)]

    if PLOT:
        fig, ax = plt.subplots(1,1, figsize=(13,4))
        ax.step(range(num_hours*60), df['Q_HP'], where='post', color='blue', alpha=0.6)
        ax.step(range(num_hours*60), df['Q_HP_expected'], where='post', color='blue', alpha=0.6, linestyle='dashed')
        ax.step(range(num_hours*60), load, where='post', color='red', alpha=0.6)
        ax.plot([0] + list(df['SOC']), color='orange', alpha=0.8)
        # ax2 = ax.twinx()
        # ax2.step(range(num_hours*60), c_el_crop, where='post', color='gray', alpha=0.6)
        # ax2.set_ylim([0,max(c_el)])
        plt.show()
    
    # Prepare for next iteration: get the intial storage
    SoC_0 = df['SOC'].iloc[-1]

    # Extend all lists for final plot
    Q_HP_list.extend(df['Q_HP'].tolist())
    Q_HP_expected_list.extend(df['Q_HP_expected'].tolist())
    load_list.extend(load)
    SOC_list.extend(df['SOC'])
    
c_el_list = [0.0359, 0.0206, 0.0106, 0.0192, 0.0309, 0.0612, 0.0925, 0.1244, 0.1667, 0.2148, 0.3563,
0.4893, 0.7098, 0.7882, 0.5586, 0.3326, 0.2152, 0.1487, 0.0864, 0.0587, 0.0385, 0.0246, 0.0165, 0.0215]
c_el_list = [x*100 for x in c_el_list]
c_el_list = c_el_list[:num_iterations]
c_el_list = [x for x in c_el_list for _ in range(60)]

# Plot
fig, ax = plt.subplots(1,1, figsize=(13,4))
ax.step(range(num_iterations*60), Q_HP_list, where='post', color='blue', alpha=0.6, label="HP real")
ax.step(range(num_iterations*60), Q_HP_expected_list, where='post', color='blue', alpha=0.6, linestyle='dotted', label="HP predicted")
ax.step(range(num_iterations*60), load_list, where='post', color='red', alpha=0.6, label="Load")
ax.plot([0] + SOC_list, color='orange', alpha=0.8, label="Storage")
ax.plot([max_storage]*num_iterations*60, color='orange', alpha=0.8, label="Maximum storage", linestyle='dashed')
ax2 = ax.twinx()
ax2.step(range(num_iterations*60), c_el_list, where='post', color='gray', alpha=0.4, label="Electricity price")

# Hours in xticks
hours = range(0, num_iterations+1, 60)
hour_labels = range(0, num_iterations+1)
ax.set_xticks(range(0, num_iterations*60+1, 60))
ax.set_xticklabels(hour_labels)

ax.set_xlabel("Time [hours]")
ax.set_ylabel("Energy [kWh]")
ax2.set_ylabel("Electricity price [cts/kWh]")

ax.set_ylim([0,25])

ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()
