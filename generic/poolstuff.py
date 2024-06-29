import pandas as pd

def get_modes(elec_costs):

    modes = {
        'Speed1': {'RPM': 750},
        'Speed2': {'RPM': 1500},
        'Speed3': {'RPM': 2350},
        'Speed4': {'RPM': 3110},
        'Max': {'RPM': 3450}
        }

    mode_map = {
        'Normal': 'Speed3',
        'LU': 'Speed4',
        'ALU': 'Max',
        'Shed': 'Speed2'
        }

    max_power = 3.2 # kW
    max_flow = 80 # gal/min

    # Assuming that min_flow=speed1, max_flow=max_rpm
    for mode in modes.keys():
        modes[mode]['flow (gal/min)'] = max_flow * modes[mode]['RPM'] / modes['Max']['RPM'] #slope * modes[mode]['RPM'] + intercept
        modes[mode]['power (kW)'] = max_power * (modes[mode]['RPM']/modes['Max']['RPM'])**3  #modes[mode]['RPM'] / modes['Max']['RPM'] * max_power

    # Read prices
    prices = pd.DataFrame({'column1':elec_costs})

    # Creating baseline
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
        baseline_operation.loc[hr, 'Flow (gal)'] = modes[mode_map[baseline_operation.loc[hr, 'Mode']]]['flow (gal/min)'] * 60
        baseline_operation.loc[hr, 'Power (kW)'] = modes[mode_map[baseline_operation.loc[hr, 'Mode']]]['power (kW)']
        baseline_operation.loc[hr, 'Cost ($)'] = baseline_operation.loc[hr, 'Power (kW)'] * prices.column1.loc[hr]

    # Creating dataframe of all operation options
    operation = pd.DataFrame(index = prices.index)
    operation['Normal'] = prices * modes[mode_map['Normal']]['power (kW)'] / modes[mode_map['Normal']]['flow (gal/min)']
    operation['LU'] = prices * modes[mode_map['LU']]['power (kW)'] / modes[mode_map['Normal']]['flow (gal/min)']
    operation['ALU'] = prices * modes[mode_map['ALU']]['power (kW)' ] / modes[mode_map['Normal']]['flow (gal/min)']
    operation['Shed'] = prices * modes[mode_map['Shed']]['power (kW)'] / modes[mode_map['Normal']]['flow (gal/min)']

    # Identify the daily pumping volume from the baseline pumping schedule
    required_flow_vol = baseline_operation['Flow (gal)'].sum()

    # Create a list of operating costs for each hour/mode combintion, ranked from lowest to highest cost
    operating_cost = pd.DataFrame(columns = ['Cost ($/gal)'])
    for key in operation.columns:
        for val in operation.index:
            operating_cost.loc['{} - {}'.format(val, key), 'Cost ($/gal)'] = operation.loc[val, key]
            
    operating_cost.reset_index(inplace=True)
    operating_cost.columns = ['hour_mode','cost_pu']
    operating_cost[['hour', 'mode']] = operating_cost['hour_mode'].str.split(' - ', expand=True)
    operating_cost['hour'] = pd.to_numeric(operating_cost['hour'])
    operating_cost = operating_cost.sort_values(by='cost_pu').reset_index()[['hour','cost_pu','mode']]

    return required_flow_vol, operating_cost


def get_mapping(hour_mode, hour, hours_ranked):

    modes = {
        'Speed1': {'RPM': 750},
        'Speed2': {'RPM': 1500},
        'Speed3': {'RPM': 2350},
        'Speed4': {'RPM': 3110},
        'Max': {'RPM': 3450}
        }

    mode_map = {
        'Normal': 'Speed3',
        'LU': 'Speed4',
        'ALU': 'Max',
        'Shed': 'Speed2'
        }
    
    max_power = 3.2 # kW
    max_flow = 80 # gal/min

    # Assuming that min_flow=speed1, max_flow=max_rpm
    for mode in modes.keys():
        modes[mode]['flow (gal/min)'] = max_flow * modes[mode]['RPM'] / modes['Max']['RPM']
        modes[mode]['power (kW)'] = max_power * (modes[mode]['RPM']/modes['Max']['RPM'])**3
    
    df = pd.DataFrame(hours_ranked)
    cost_pu = df[(df['hour'] == hour) & (df['mode'] == hour_mode)]
    cost_pu = cost_pu.iloc[0]['cost_pu']

    flow = modes[mode_map[hour_mode]]['flow (gal/min)'] * 60
    cost = flow * cost_pu

    return flow, cost