import pandas as pd
import os
from functions import generic
from functions import get_storage
from functions import iteration_plot
import numpy as np

'''
******************************
OPEN LOOP EXAMPLE
******************************
'''
print('*'*30+'\nOPEN LOOP\n'+'*'*30)

parameters = { 

    # Prediction horizon, in hours
    'horizon': 24,

    # Electricity prices for every hour in the horizon, in cts/kWh
    'elec_costs': [7.92, 6.63, 6.31, 6.79, 8.01, 11.58, 19.38, 21.59, 11.08, 4.49, 1.52, 0.74, 
                   0.42, 0.71, 0.97, 2.45, 3.79, 9.56, 20.51, 28.26, 23.49, 18.42, 13.23, 10.17],
    
    # The load can either be hourly or daily
    # When hourly, the values are given for every hour in the horizon, in kWh
    # When daily, a single value is given in kWh
    'load': {'type': 'hourly', 
             'value': [5.91, 5.77, 5.67, 5.77, 5.71, 6.06, 6.34, 6.34, 6.01, 5.77, 5.05, 5.05, 
                       4.91, 4.91, 4.91, 4.91, 5.05, 5.1, 4.91, 4.91, 4.91, 4.91, 4.98, 4.91]},

    # With a HP, the control is the hourly heating power (kW or kWh, both are equivalent with hourly time steps)
    # The control can be within a continuous operating range (i.e. it must be between a given minimum and a maximum value)
    # Or the control can be a choice between discrete operating modes
    'control': {'type': 'range',
                'max': [12]*24, #kW
                'min': [6]*24}, #kW
    
    # The constraints are activated using True instead of false, and have corresponding parameters
    'constraints': {'storage_capacity': True,   # If there is a storage limit
                    'max_storage': 30,          # The storage capacity, in kWh
                    'initial_soc': 0,           # Initial state of charge
                    'cheaper_hours': True,      # Leave True when working with hourly loads
                    'quiet_hours': False        # Assign hours during which the system must not operate
                    },

    'hardware': {'heatpump': True,              # True if there is a heat pump
                 'COP': [3]*24}                 # Estimated COP at every hour of the horizon, typically COP(T_OA)
}

# Obtain the control sequence from Pareto
control = generic(parameters)
print(f"\nThe control sequence is:\n{control} kWh\n")

'''
******************************
CLOSED LOOP EXAMPLE
******************************
'''

print('*'*30+'\nCLOSED LOOP\n'+'*'*30+'\n')

# Initialize
final_control_sequence = [0]*24

# Collecting yearly data (load, outside air temperature, electricity prices)
df = pd.read_excel(os.getcwd()+'/data/gridworks_yearly_data.xlsx', header=3, index_col = 0)
df.index = pd.to_datetime(df.index)
df.index.name = None
df['elec'] = df['Total Delivered Energy Cost ($/MWh)']
df['load'] = df['House Power Required AvgKw'] *0.8
df['T_OA'] = df['Outside Temp F']
df['T_OA'] = df['T_OA'].apply(lambda x: round(5/9 * (x-32),2))
df = df[['elec', 'load', 'T_OA']]
df.head()

# Initial state of charge (kWh)
soc_0 = 0
soc = soc_0

# Initialize cost of operating the system
total_cost = 0

# Simulating 2 days
for hour in range(24*1):

    parameters = { 
        'horizon': 24,

        'elec_costs': list(df.elec[hour:hour+24]),
    
        'load': {'type': 'hourly', 
                'value': list(df.load[hour:hour+24])},

        'control': {'type': 'range',
                    'max': [12]*24,
                    'min': [6]*24},
        
        'constraints': {'storage_capacity': True,
                        'max_storage': 30,
                        'initial_soc': soc,
                        'cheaper_hours': True,
                        'quiet_hours': False
                        },

        'hardware': {'heatpump': True,
                     'COP': [3]*24}
    }

    # Get the controls, update the storage
    control = generic(parameters)
    final_control_sequence[hour] = control[0]

    # Get the storage level (FMU or compute) and update SoC for next iteration
    storage = get_storage(control, parameters)
    soc = storage[1] if storage[1]<parameters['constraints']['max_storage'] else parameters['constraints']['max_storage'] 

    # Cost of the last hour
    cost = control[0] * parameters['elec_costs'][0] / parameters['hardware']['COP'][0]
    total_cost += cost

    print(f'Hour {hour}, Q_HP {control[0]}, cost {round(cost,3)}, SoC {round(soc,2)}')


# Plot the whole operation
parameters['elec_costs'] = list(df.elec[:24])
parameters['load']['value'] = list(df.load[:24])
parameters['constraints']['initial_soc'] = soc_0
print(f"\nThe control sequence is:\n{final_control_sequence} kWh\n")
iteration_plot({'control': final_control_sequence}, parameters)