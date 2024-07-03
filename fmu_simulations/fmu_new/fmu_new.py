import pandas as pd
import os
from functions import generic
from functions import get_storage
from functions import iteration_plot
import statsmodels.formula.api as smf
import numpy as np
import time
import subprocess
try:
    import pyfmi
except ImportError:
    print("\nPyFMI is not installed on this device.\n")

# --------------------------
# Parameters
# --------------------------

storage_capacity = 10.3 # kWh
hp_capacity = 10        # kW
m_HP = 0.29             # kg/s
T_HP_in = 57            # Celcius
PRINT = True

# --------------------------
# Other
# --------------------------

# Yearly data
df = pd.read_excel(os.getcwd()+'/data/gridworks_yearly_data.xlsx', header=3, index_col = 0)
df.index = pd.to_datetime(df.index)
df.index.name = None
df['elec'] = df['Total Delivered Energy Cost ($/MWh)']
df['load'] = df['House Power Required AvgKw'] * 0.8
df['T_OA'] = df['Outside Temp F']
df['T_OA'] = df['T_OA'].apply(lambda x: round(5/9 * (x-32),2))
df = df[['elec', 'load', 'T_OA']]

# Get COP regression as a function of T_OA
T_OA_table = list(range(-15,40,5))
COP_table = 1.7,2.3,2.3,2.43,2.63,2.78,3.36,3.79,3.99,4.12,5.10
COP_df = pd.DataFrame({'temp':T_OA_table,'cop':COP_table})
mod = smf.ols(formula='cop ~ temp', data=COP_df)
np.random.seed(2) 
res = mod.fit()

def get_COP(T):
    return round(res.params.Intercept + T*res.params.temp, 2)

def get_T_sup_HP(Q):
    return round(Q*1000/m_HP/4187 + T_HP_in, 2)

# --------------------------
# Simulate one hour
# --------------------------

def simulate(delta_HP, T_sup_HP, hour):

    # Simulate 1 hour
    start_time, final_time = 0, 3600
    
    # Inputs over the hour
    inputs_dict = {
        'HeatPumpWaterSupplyMassFlow': 0.29 * delta_HP,
        'HeatPumpWaterTempSetpoint': T_sup_HP + 273,
        'HeatPumpOnOff': delta_HP,
        'HeatPumpMode': True,
        'OutdoorAirTemperature': df.T_OA[hour] + 273,
        'ZoneHeatingLoad': df.load[hour] * 1000,
    }  

    # Buildig the FMU input
    input_names = list(inputs_dict)
    inputs_array = np.array([[minute]+list(inputs_dict.values()) for minute in range(start_time,final_time+60,60)])
    inputs = (input_names, inputs_array)
    if PRINT: print(inputs)
    
    # Simulation options
    opts = model.simulate_options()
    opts["ncp"] = 60
    if hour>0:
        state = model.get_fmu_state()
        opts['initialize'] = False
        model.set_fmu_state(state)

    # Simulate 1 hour
    res = model.simulate(start_time=start_time, final_time=final_time, input=inputs, options=opts)
    if PRINT: print(f"\nThe simulation has finished running on the FMU.")

    # The simulation outputs a .mat file with results, convert it to csv
    time.sleep(1)
    command = f"python mat_to_csv.py {fmuName.replace(".fmu","")+'_result.mat'} > prints.txt"
    subprocess.call(command, shell=True)

    # Read and return the results file as a dataframe
    results_df = pd.read_csv(fmuName.replace(".fmu","")+'_result.csv').drop('Unnamed: 0', axis=1)
    return results_df

# --------------------------
# Initialize
# --------------------------

# Control sequence
final_Q_HP_sequence = []

# State of charge (kWh)
soc_0 = 0
soc = soc_0

# Initialize cost of operating the system
total_cost = 0

# Load FMU
fmuName = 'R32SimpleHpTesDummyZone20kWh.fmu'
model = pyfmi.load_fmu(fmuName)
model.set('phaseChangeBattery58.Design.Tes_nominal', storage_capacity*3600000)
if PRINT: print(f"Model {fmuName} was loaded.\n")

# --------------------------
# Simulating 24 hours
# --------------------------

print('*'*30+'\nFMU closed loop simulation\n'+'*'*30+'\n')

for hour in range(24):

    parameters = { 
        'horizon': 24,

        'elec_costs': list(df.elec[hour:hour+24]),
    
        'load': {'type': 'hourly', 
                'value': list(df.load[hour:hour+24])},

        'control': {'type': 'range',
                    'max': [hp_capacity]*24,
                    'min': [0.2*hp_capacity]*24},
        
        'constraints': {'storage_capacity': True,
                        'max_storage': storage_capacity,
                        'initial_soc': soc,
                        'cheaper_hours': True,
                        'quiet_hours': False},

        'hardware': {'heatpump': True,
                     'COP': [get_COP(x) for x in df.T_OA[hour:hour+24]]}
    }

    # Get the controls from the generic algorithm
    Q_HP = generic(parameters)
    final_Q_HP_sequence.append(Q_HP[0])

    # Convert to temperature setpoint and delta
    delta_HP = 1 if Q_HP[0]>0 else 0
    T_sup_HP = get_T_sup_HP(Q_HP[0]) if Q_HP[0]>0 else 0

    # Send commands to FMU and obtain simulation results
    df = simulate(delta_HP, T_sup_HP, hour)
    df = df.drop(df.index[-1])
    print(df)

    # Convert SoC to kWh and update
    soc = round(df['SOC'].iloc[-1] * storage_capacity, 2)

    # Cost of the last hour
    cost = Q_HP[0] * parameters['elec_costs'][0] / parameters['hardware']['COP'][0]
    total_cost += cost

    print(f'Hour {hour}, Q_HP {Q_HP[0]}, cost {round(cost,3)}, SoC {round(soc,2)}')

# Plot the whole operation
parameters['elec_costs'] = list(df.elec[:24])
parameters['load']['value'] = list(df.load[:24])
parameters['constraints']['initial_soc'] = soc_0
print(f"\nThe control sequence is:\n{final_Q_HP_sequence} kWh\n")
iteration_plot({'control': final_Q_HP_sequence}, parameters)