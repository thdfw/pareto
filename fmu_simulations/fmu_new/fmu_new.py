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

storage_capacity = 20   # kWh
hp_capacity = 12        # kW
m_HP = 0.29             # kg/s
T_HP_in = 57            # Celcius
PRINT = True

# --------------------------
# Other
# --------------------------

# Yearly data
df_yearly = pd.read_excel(os.getcwd()+'/data/gridworks_yearly_data.xlsx', header=3, index_col = 0)
df_yearly.index = pd.to_datetime(df_yearly.index)
df_yearly.index.name = None
df_yearly['elec'] = df_yearly['Total Delivered Energy Cost ($/MWh)']
df_yearly['load'] = df_yearly['House Power Required AvgKw'] * 0.8
df_yearly['T_OA'] = df_yearly['Outside Temp F']
df_yearly['T_OA'] = df_yearly['T_OA'].apply(lambda x: round(5/9 * (x-32),2))
df_yearly = df_yearly[['elec', 'load', 'T_OA']]

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
        'OutdoorAirTemperature': df_yearly.T_OA[hour] + 273,
        'ZoneHeatingLoad': df_yearly.load[hour] * 1000,
    }  

    # Buildig the FMU input
    input_names = list(inputs_dict)
    inputs_array = np.array([[minute]+list(inputs_dict.values()) for minute in range(start_time,final_time+60,60)])
    inputs = (input_names, inputs_array)
    
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
    command = f"python mat_to_csv.py {fmuNameNoSuffix+'_result.mat'} > prints.txt"
    subprocess.call(command, shell=True)

    # Read and return the results file as a dataframe
    df = pd.read_csv(fmuNameNoSuffix+'_result.csv').drop('Unnamed: 0', axis=1)
    return df

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
fmuNameNoSuffix = fmuName.replace(".fmu","")
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

        'elec_costs': list(df_yearly.elec[hour:hour+24]),
    
        'load': {'type': 'hourly', 
                'value': list(df_yearly.load[hour:hour+24])},

        'control': {'type': 'range',
                    'max': [hp_capacity]*24,
                    'min': [0.2*hp_capacity]*24},
        
        'constraints': {'storage_capacity': True,
                        'max_storage': storage_capacity,
                        'initial_soc': soc,
                        'cheaper_hours': True,
                        'quiet_hours': False},

        'hardware': {'heatpump': True,
                     'COP': [get_COP(x) for x in df_yearly.T_OA[hour:hour+24]]}
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

    # Convert SOC to kWh
    df['SOC'] = df['SOC'] * storage_capacity
    df['SOC'] = df['SOC'].round(2)
    # Compute real Q_HP
    df['Q_HP'] = df['HeatPumpOnOff'] * 0.29 * 4187 * (df['T_HP_sup'] - df['T_HP_ret']) / 1000
    df['Q_HP'] = df['Q_HP'].round(2)
    df['HeatPumpOnOff'] = df['HeatPumpOnOff'].round()
    df['T_HP_ret'] = df['T_HP_ret'].round(1)
    df['T_HP_sup'] = df['T_HP_sup'].round(1)
    # Add inputs to df
    df['INPUT_delta_HP'] = [delta_HP for _ in range(60)]
    df['INPUT_T_HP_sup_setpoint'] = [T_sup_HP+273 for _ in range(60)]
    # Compute expected Q_HP
    df['Q_HP_expected'] = df['INPUT_delta_HP'] * 0.29 * 4187 * (df['INPUT_T_HP_sup_setpoint'] - T_HP_in - 273) / 1000
    df['Q_HP_expected'] = df['Q_HP_expected'].round(2)
    print(df)

    # Update SoC
    soc = df['SOC'].iloc[-1]

    # Cost of the last hour
    cost = Q_HP[0] * parameters['elec_costs'][0] / parameters['hardware']['COP'][0]
    total_cost += cost

    print(f'Hour {hour}, Q_HP {Q_HP[0]}, cost {round(cost,3)}, SoC {round(soc,2)}')

# Plot the whole operation
parameters['elec_costs'] = list(df_yearly.elec[:24])
parameters['load']['value'] = list(df_yearly.load[:24])
parameters['constraints']['initial_soc'] = soc_0
print(f"\nThe control sequence is:\n{final_Q_HP_sequence} kWh\n")
iteration_plot({'control': final_Q_HP_sequence}, parameters)