import pandas as pd
import os
from functions import generic
from functions import iteration_plot
import statsmodels.formula.api as smf
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess
try:
    import pyfmi
except ImportError:
    print("\nPyFMI is not installed on this device.\n")

# --------------------------
# Parameters
# --------------------------

storage_capacity = 10.3 # kWh
hp_capacity = 10.35 # kW
m_HP = 0.29 # kg/s
PRINT = False

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

# Use CalFlexHub prices
df_yearly['elec'] = [0.1714, 0.144, 0.1385, 0.1518, 0.1829, 0.2713, 0.4659, 0.5328, 0.28, 0.1158, 
                     0.0398, 0.0196, 0.011, 0.0188, 0.0255, 0.0632, 0.0957, 0.2358, 0.4931, 0.6618, 
                     0.5364, 0.4116, 0.2905, 0.2209]*365

# Get COP regression as a function of T_OA
T_OA_table = list(range(-15,40,5))
COP_table = 1.7,2.3,2.3,2.43,2.63,2.78,3.36,3.79,3.99,4.12,5.10
COP_df = pd.DataFrame({'temp':T_OA_table,'cop':COP_table})
mod = smf.ols(formula='cop ~ temp', data=COP_df)
np.random.seed(2) 
res = mod.fit()

def get_COP(T):
    return round(res.params.Intercept + T*res.params.temp, 2)

def get_T_HP_in(soc):
    print(f'T_HP_in={56.35 + 0.154*soc}')
    return 55.8 + (0.3*soc if soc/storage_capacity<0.5 else 0.6*soc)

def get_T_sup_HP(Q, soc):
    return round(Q*1000/m_HP/4187 + get_T_HP_in(soc), 2)

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

# Lists for final plot and analysis
Q_HP_list = []
Q_HP_expected_list = []
load_list = []
SOC_list = []
T_ret_list = []

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
    T_sup_HP = get_T_sup_HP(Q_HP[0], soc) if Q_HP[0]>0 else 0

    # Send commands to FMU and obtain simulation results
    df = simulate(delta_HP, T_sup_HP, hour)
    df = df.drop(df.index[-1])

    df['SOC'] = df['SOC'] * storage_capacity
    df['SOC'] = df['SOC'].round(2)
    df['HeatPumpOnOff'] = df['HeatPumpOnOff'].round()
    df['Q_HP'] = df['HeatPumpOnOff'] * 0.29 * 4187 * (df['T_HP_sup'] - df['T_HP_ret']) / 1000
    df['Q_HP'] = df['Q_HP'].round(2)
    df['T_HP_ret'] = df['T_HP_ret'].round(1) - 273
    df['T_HP_sup'] = df['T_HP_sup'].round(1) - 273
    df['T_HP_sup_setpoint'] = [T_sup_HP for _ in range(60)]
    df['Q_HP_expected'] = [Q_HP[0]] * len(df)
    df['Q_HP_expected'] = df['Q_HP_expected'].round(2)
    if PRINT: print(df[['T_HP_sup_setpoint', 'T_HP_sup','T_HP_ret','Q_HP','Q_HP_expected']])

    Q_HP_list.extend(list(df['Q_HP']))
    Q_HP_expected_list.extend(list(df['Q_HP_expected']))
    load_list.extend([df_yearly.load[hour] for _ in range(60)])
    SOC_list.extend(list(df['SOC']))
    T_ret_list.extend(df.T_HP_ret)

    # Update SoC
    soc = df['SOC'].iloc[-1]
    soc = storage_capacity if soc>storage_capacity else soc

    # Cost of the last hour
    cost = Q_HP[0] * parameters['elec_costs'][0] / parameters['hardware']['COP'][0]
    total_cost += cost

    print('*'*30+f'\nHour {hour}, Q_HP {Q_HP[0]}, cost {round(cost,3)}, SoC {round(soc,2)}\n'+'*'*30)

# --------------------------
# Plot
# --------------------------

print('\n\nHello\n\n')

T_ret_df = pd.DataFrame({'soc':SOC_list,'t_ret':T_ret_list})
T_ret_df = T_ret_df[60:]
print(T_ret_df)
mod = smf.ols(formula='t_ret ~ soc', data=T_ret_df)
np.random.seed(2) 
res = mod.fit()
print('done')
print(res.params.Intercept)
print(res.params.soc)
print('done')

# --------------------------
# Plot
# --------------------------

c_el_list = df_yearly.elec[:24]
c_el_list = [x for x in c_el_list for _ in range(60)]

SOC_list = [soc_0] + SOC_list
SOC_list_percent = [x/storage_capacity*100 for x in SOC_list]
SOC_list_percent = [x if x<100 else 100 for x in SOC_list]

fig, ax = plt.subplots(2,1, figsize=(8,5), sharex=True)
ax[0].step(range(24*60), Q_HP_list, where='post', color='blue', alpha=0.6, label="Heat pump")
ax[0].step(range(24*60), Q_HP_expected_list, where='post', color='blue', alpha=0.6, linestyle='dotted', label="Objective")
ax[0].step(range(24*60), load_list, where='post', color='red', alpha=0.6, label="Load")
ax[1].plot(SOC_list_percent, color='orange', alpha=0.8, label="Storage")
#ax[1].plot([storage_capacity]*24*60, color='orange', alpha=0.8, label="Maximum storage", linestyle='dashed')
ax2 = ax[0].twinx()
ax2.step(range(24*60), c_el_list, where='post', color='gray', alpha=0.4, label="Electricity price")

hours = range(0, 24+1, 60)
hour_labels = range(0, 24+1)
ax[0].set_xticks(range(0, 24*60+1, 60))
ax[0].set_xticklabels(hour_labels)

ax[1].set_xlabel("Time [hours]")
ax[0].set_ylabel("Power [kW]")
ax[1].set_ylabel("State of charge [%]")
ax2.set_ylabel("Price [cts/kWh]")

ax[0].set_ylim([0,hp_capacity+10])

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()