import pandas as pd
import os
from functions import baseline
from functions import get_storage
from functions import iteration_plot
import numpy as np

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
                  0.5364, 0.4116, 0.2905, 0.2209]*31

N = 12

parameters = { 

    'horizon': N,

    'elec_costs': CFH_prices[:N],
    
    'load': {'type': 'hourly', 
             'value': list(df.load[:N])},

    'control': {'type': 'range',
                'max': [12]*N, #kW
                'min': [6]*N}, #kW
    
    'constraints': {'storage_capacity': True,
                    'max_storage': 10,
                    'initial_soc': 0,
                    'cheaper_hours': True,
                    'quiet_hours': False
                    },

    'hardware': {'heatpump': True,
                 'COP': [3]*N}
}

# Obtain the control sequence from Pareto
control = baseline(parameters)
print(f"\nThe control sequence is:\n{control} kWh\n")

# Plot the whole operation
iteration_plot({'control': control}, parameters)