# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:11:29 2024

@author: Peter Grant
"""

# Could do this differently. We don't need to know pumped gallons. Only revolutions of the motor
# Can do analysis based on solely on completing a certain number of revolutions
# Notes from EPRI/NREL report: (Use these notes to update algorithm)
    # - Shed reduces rpm to 1/3 of default speed
    # - Load up increases to maximum, which is set by the user. Default max is 3000 rpm. Can be adjusted as high as 3450 rpm

import pandas as pd

# reference: https://www.pentair.com/content/dam/extranet/nam/pentair-pool/residential/pumps/archive/intelliflo-vsf/intelliflo-vsf-om-eng.pdf
modes = {
         'Speed1': {
                    'RPM': 750
                   },
         'Speed2': {
                    'RPM': 1500
                   },
         'Speed3': {
                    'RPM': 2350
                   },
         'Speed4': {
                    'RPM': 3110,
                   },
         'Max': {
                 'RPM': 3450
                }
    }

mode_map = {
            'Normal': 'Speed3',
            'LU': 'Speed4',
            'ALU': 'Max',
            'Shed': 'Speed2'
           }

# reference: https://engineeringlibrary.org/reference/centrifugal-pumps-fluid-flow-doe-handbook#:~:text=These%20laws%20state%20that%20the,cube%20of%20the%20pump%20speed.
# volumetric_flowrate is proportional to rpm
# pump power is proportional to rpm^3

# reference: https://www.pentair.com/content/dam/extranet/nam/pentair-pool/residential/pumps/archive/intelliflo-vsf/intelliflo-vsf-om-eng.pdf
max_power = 3.2 # kW
# min_flow = 30 # gal/min
max_flow = 80 # gal/min
# slope = (max_flow - min_flow) / (modes['Max']['RPM'] - modes['Speed1']['RPM'])
# intercept = min_flow - modes['Speed1']['RPM'] * slope

# assuming that min flow = speed1, max flow = max rpm
# Would need to verify this assumption with Pentair for field deployment, but good enough for now

for mode in modes.keys():
    modes[mode]['flow (gal/min)'] = max_flow * modes[mode]['RPM'] / modes['Max']['RPM'] #slope * modes[mode]['RPM'] + intercept
    modes[mode]['power (kW)'] = max_power * (modes[mode]['RPM']/modes['Max']['RPM'])**3  #modes[mode]['RPM'] / modes['Max']['RPM'] * max_power

# assume occpants want low noise, low flow rates overnight
low_flow_hours = [0, 1, 2, 3, 4, 5, 6, 22, 23]

# read prices
price_structure = 'SummerHDP'
prices = pd.read_csv('CFH_Prices.csv', index_col = 0).loc[0:23, price_structure]

# creating baseline
# assumptions based on https://efficiencymb.ca/articles/programming-your-variable-speed-pool-pump/#:~:text=Scheduling%20your%20pump%20to%20run,1%2C500%20RPM%20and%202%2C000%20RPM.&text=You'll%20want%20to%20run,to%20four%20hours%20every%20day.
# assuming 3 hrs of high speed (Speed4), 20 hours of low speed (Speed2)
# assuming high speed in the middle of the day when people will be less irritated by noise
baseline_operation = pd.DataFrame(index = prices.index, columns = ['Mode', 'Flow (gal)', 'Power (kW)', 'Cost ($)'])
baseline_operation['Flow (gal)'] = float(0)
baseline_operation['Power (kW)'] = float(0)
baseline_operation['Cost ($)'] = float(0)
for hr in baseline_operation.index:
    if hr == 0 or hr == 1 or hr == 2 or hr == 3 or hr == 4 or hr == 5 or hr == 6 or hr == 7 or hr == 21 or hr == 22 or hr == 23:
        baseline_operation.loc[hr, 'Mode'] = 'Off'
        continue
    elif hr == 8 or hr == 9 or hr == 10:
        baseline_operation.loc[hr, 'Mode'] = 'LU'
    else:
        baseline_operation.loc[hr, 'Mode'] = 'Normal'
    
    # cost = prices.loc[hr] * modes['Speed3']['power (kW)']
    # baseline_operation.loc[hr, 'Mode'] = 'Normal'
    baseline_operation.loc[hr, 'Flow (gal)'] = modes[mode_map[baseline_operation.loc[hr, 'Mode']]]['flow (gal/min)'] * 60
    baseline_operation.loc[hr, 'Power (kW)'] = modes[mode_map[baseline_operation.loc[hr, 'Mode']]]['power (kW)']
    baseline_operation.loc[hr, 'Cost ($)'] = baseline_operation.loc[hr, 'Power (kW)'] * prices.loc[hr]

# Creating dataframe of all operation options
operation = pd.DataFrame(index = prices.index)
operation['Normal'] = prices * modes[mode_map['Normal']]['power (kW)'] / modes[mode_map['Normal']]['flow (gal/min)']
operation['LU'] = prices * modes[mode_map['LU']]['power (kW)'] / modes[mode_map['Normal']]['flow (gal/min)']
operation['ALU'] = prices * modes[mode_map['ALU']]['power (kW)' ] / modes[mode_map['Normal']]['flow (gal/min)']
operation['Shed'] = prices * modes[mode_map['Shed']]['power (kW)'] / modes[mode_map['Normal']]['flow (gal/min)']

# identify the daily pumping volume from the baseline pumping schedule
required_flow_vol = baseline_operation['Flow (gal)'].sum()

# create a list of operating costs for each hour/mode combintion, ranked from lowest to highest cost
operating_cost = pd.DataFrame(columns = ['Cost ($/gal)'])
# ix = []
for key in operation.columns:
    for val in operation.index:
        operating_cost.loc['{} - {}'.format(val, key), 'Cost ($/gal)'] = operation.loc[val, key]
        
operating_cost = operating_cost['Cost ($/gal)'].sort_values()
print(operating_cost)
# assumption: all pump modes are maintained for 1 hour. We may want to change that assumption

# identify the lowest cost operating schedule that meets the daily pumping volume
pumped_vol = 0
# create dataframe to show results
operation_times = pd.DataFrame(index = prices.index, columns = ['Mode', 'Flow (gal)', 'Power (kW)', 'Cost ($)'])
operation_times['Flow (gal)'] = float(0)
operation_times['Power (kW)'] = float(0)
operation_times['Cost ($)'] = float(0)
while operation_times['Flow (gal)'].sum() < required_flow_vol: # until we have identified adequate pumping
    operation_time = operating_cost.index[0].split(' - ')[0] # gather time of lowest cost operation
    operation_mode = operating_cost.index[0].split(' - ')[1] # gather mode of lowest cost operation
    if int(operation_time) in low_flow_hours: # avoid operation choices that conflict with user-specified "quiet operation" times
        if operation_mode != 'Shed':
            operating_cost.drop(operating_cost.index[0], inplace = True)
            continue
    # identify the impacts of the cheapest operating mode
    flow_vol = modes[mode_map[operation_mode]]['flow (gal/min)'] * 60
    operation_times.loc[int(operation_time), 'Mode'] = operation_mode
    operation_times.loc[int(operation_time), 'Flow (gal)'] = flow_vol
    operation_times.loc[int(operation_time), 'Power (kW)'] = modes[mode_map[operation_mode]]['power (kW)']
    operation_times.loc[int(operation_time), 'Cost ($)'] = operation_times.loc[int(operation_time), 'Power (kW)'] * prices.loc[int(operation_time)]
    
    # remove that operating mode from the list, moving to the next cheapest option in the next iteration
    operating_cost.drop(operating_cost.index[0], inplace = True)

supervised_cost = operation_times['Cost ($)'].sum()
baseline_cost = baseline_operation['Cost ($)'].sum()
savings = baseline_cost - supervised_cost
savings_percent = savings / baseline_cost * 100
print('supervised_cost: {}'.format(supervised_cost))
print('baseline_cost: {}'.format(baseline_cost))
print('savings: {}'.format(savings))
print('savings_percent: {}'.format(savings_percent))

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (12, 6))
plt.plot(operation_times['Flow (gal)'], label = 'Price Responsive')
plt.plot(baseline_operation['Flow (gal)'], label = 'Standard')
plt.ylabel('Flow (gal/hr)')

fig = plt.figure(figsize = (12, 6))
plt.plot(operation_times['Power (kW)'], label = 'Price Responsive')
plt.plot(baseline_operation['Power (kW)'], label = 'Standard')
plt.ylabel('Electric Demand (kW)')

fig = plt.figure(figsize = (12, 6))
plt.plot(operation_times['Cost ($)'], label = 'Price Responsive')
plt.plot(baseline_operation['Cost ($)'], label = 'Standard')
plt.ylabel('Operating Cost ($/hr)')

fig = plt.figure(figsize = (12, 6))
plt.plot(operation_times['Cost ($)'].cumsum(), label = 'Price Responsive')
plt.plot(baseline_operation['Cost ($)'].cumsum(), label = 'Standard')
plt.ylabel('Cumulative Operating Cost ($)')

# assumption: 1.25 million pools in CA. https://www.politico.com/newsletters/california-climate/2023/10/18/california-is-coming-for-your-pools-00122374#:~:text=There%20are%201.25%20million%20residential,because%20their%20pumps%20are%20bigger.
fleet_size = 1250000

# plotting impacts of entire fleet
fig = plt.figure(figsize = (12, 6))
plt.plot(operation_times['Power (kW)'] * fleet_size, label = 'Price Responsive')
plt.plot(baseline_operation['Power (kW)'] * fleet_size, label = 'Standard')
plt.ylabel('Electric Demand (kW)')

fig = plt.figure(figsize = (12, 6))
plt.plot(operation_times['Cost ($)'] * fleet_size, label = 'Price Responsive')
plt.plot(baseline_operation['Cost ($)'] * fleet_size, label = 'Standard')
plt.ylabel('Operating Cost ($/hr)')

fig = plt.figure(figsize = (12, 6))
plt.plot(operation_times['Cost ($)'].cumsum() * fleet_size, label = 'Price Responsive')
plt.plot(baseline_operation['Cost ($)'].cumsum() * fleet_size, label = 'Standard')
plt.ylabel('Cumulative Operating Cost ($)')

print('fleet supervised_cost: {}'.format(supervised_cost * fleet_size))
print('fleet baseline_cost: {}'.format(baseline_cost * fleet_size))
print('fleet savings: {}'.format(savings * fleet_size))