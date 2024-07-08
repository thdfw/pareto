import pandas as pd
import os
from functions import iteration_plot
import numpy as np
import statsmodels.formula.api as smf

# -----------------------------------------
# Baseline algorithm
# -----------------------------------------

def baseline(parameters):

    load = parameters['load']['value']
    storage_capacity = parameters['constraints']['max_storage']
    controls = []

    peak_hours = [6,7,19,20]
    pre_morning_hours = [0,1,2,3,4,5]
    pre_evening_hours = [8,9,10,11,12,13,14,15,16,17,18]

    needed_storage_morning = round(sum(load[6:8]),3)
    needed_storage_evening = round(sum(load[19:21]),3)

    if needed_storage_morning > storage_capacity:
        peak_hours.remove(6)
        needed_storage_morning = round(load[6])
    if needed_storage_evening > storage_capacity:
        peak_hours.remove(19)
        needed_storage_evening = round(load[19])

    for time in range(len(load)):

        hour = time%24

        if hour in peak_hours:
            controls.append(0)

        elif hour in pre_morning_hours:
            controls.append(load[hour]+needed_storage_morning/len(pre_morning_hours))

        elif hour in pre_evening_hours:
            controls.append(load[hour]+needed_storage_evening/len(pre_evening_hours))

        else:
            controls.append(load[hour])

    cost = 0
    for hour in range(parameters['horizon']):
        cost += controls[hour] * parameters['elec_costs'][hour] / parameters['hardware']['COP'][hour]

    return controls, cost

# -----------------------------------------
# COP(T_OA) regression
# -----------------------------------------

T_OA_table = list(range(-15,40,5))
COP_table = 1.7,2.3,2.3,2.43,2.63,2.78,3.36,3.79,3.99,4.12,5.10
COP_df = pd.DataFrame({'temp':T_OA_table,'cop':COP_table})
mod = smf.ols(formula='cop ~ temp', data=COP_df)
np.random.seed(2) 
res = mod.fit()
def COP(T_OA):
    return round(res.params.Intercept+T_OA*res.params.temp,2)

# -----------------------------------------
# Temperature, load and electricity data
# -----------------------------------------

df = pd.read_excel(os.getcwd()+'/data/gridworks_yearly_data.xlsx', header=3, index_col = 0)
df.index = pd.to_datetime(df.index)
df.index.name = None
df['elec'] = df['Total Delivered Energy Cost ($/MWh)']
df['load'] = df['House Power Required AvgKw'] *0.8
df['T_OA'] = df['Outside Temp F']
df['T_OA'] = df['T_OA'].apply(lambda x: round(5/9 * (x-32),2))
df = df[['elec', 'load', 'T_OA']]

CFH_prices = [0.1714, 0.144, 0.1385, 0.1518, 0.1829, 0.2713, 0.4659, 0.5328, 0.28, 0.1158, 
                  0.0398, 0.0196, 0.011, 0.0188, 0.0255, 0.0632, 0.0957, 0.2358, 0.4931, 0.6618, 
                  0.5364, 0.4116, 0.2905, 0.2209]

# -----------------------------------------
# Simulate several days
# -----------------------------------------

total_cost = 0
num_days = 3

print(f'\n--- Simulating {num_days} days ---\n')
for day in range(num_days):

    parameters = { 

        'horizon': 24,

        'elec_costs': CFH_prices,
        
        'load': {'type': 'hourly', 
                'value': list(df.load[day*24:24+day*24])},

        'control': {'type': 'range',
                    'max': [12]*24, #kW
                    'min': [6]*24}, #kW
        
        'constraints': {'storage_capacity': True,
                        'max_storage': 10,
                        'initial_soc': 0,
                        'cheaper_hours': True,
                        'quiet_hours': False
                        },

        'hardware': {'heatpump': True,
                    'COP': [COP(T) for T in list(df.T_OA[day*24:24+day*24])]}
    }

    # Obtain the control sequence from Pareto
    control, cost = baseline(parameters)
    print(f"The cost of this sequence is: {round(cost,2)} $")

    # Plot the whole operation
    iteration_plot({'control': control}, parameters)

    total_cost += cost

print(f'\nThe total cost is {round(total_cost,3)} $\n')

