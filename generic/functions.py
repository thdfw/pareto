import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from poolstuff import get_mapping

# Choose the desired level of output detail
INTERMEDIATE_PLOT = False
FINAL_PLOT = False
PRINT = False

def generic(parameters):
    """
    Computes a near-optimal sequence of operation 
    based on the given forecasts, contraints, and other parameters
    """

    # Check that parameters are coherent
    check_parameters(parameters)

    # ---------------------------------------------------
    # Initialize
    # ---------------------------------------------------

    # The solution vector
    operation = {
        'control': [0]*parameters['horizon'],
        'control_max': parameters['control']['max'] if parameters['control']['type']=='range' else [np.nan]*parameters['horizon'],
        'control_min': parameters['control']['min'] if parameters['control']['type']=='range' else [np.nan]*parameters['horizon'],
        'mode': [np.nan]*parameters['horizon'],
        'cost': [0]*parameters['horizon']
        }

    # Hours (with associated modes) ranked by price
    hours_ranked = get_hours_ranked(parameters)
    
    # Status vector
    if parameters['load']['type'] == 'hourly':
        status = [1]*parameters['horizon']
    elif parameters['load']['type'] == 'daily':
        status = [0]*(parameters['horizon']-1) + [1]
    if parameters['constraints']['storage_capacity']:
        status = get_status(operation['control'], parameters)
        if sum(status)==0:
            return operation['control']
        
    iteration_count = 0

    # ---------------------------------------------------
    # Run the generic algorithm
    # ---------------------------------------------------

    # While all hours are not satisfied
    while sum(status) > 0:

        # Use maximum power if unable to find a solution
        iteration_count += 1
        if iteration_count > 50:
            if PRINT: print('Could not converge!')
            return [operation['control_max'][0]] + [0]*(parameters['horizon']-1)

        # Find the next unsatisfied hour
        next_unsatisfied = status.index(1)
        if PRINT and parameters['load']['type']=='hourly': 
            print("\n---------------------------------------")
            print(f"The next unsatisfied hour: {next_unsatisfied}:00")
            print("---------------------------------------")

        rank = -1

        # Find and turn on the cheapest remaining hour(s) before it, until it is satisfied
        for hour in hours_ranked['hour']:
            
            # The price ranking of the hour
            rank += 1
            
            # Skip hours that are after the next unsatisfied hour
            if hour > next_unsatisfied: 
                continue

            # Skip hours which are already used to their maximum
            if parameters['control']['type']=='range' and operation['control'][hour]==operation['control_max'][hour]: 
                continue
            if parameters['control']['type']=='mode' and operation['mode'][hour]=='ALU': 
                continue

            # Skip hours before a maximum storage occurence
            if parameters['constraints']['storage_capacity']:
                storage_full = [1 if round(x,1)==parameters['constraints']['max_storage'] else 0 for x in get_storage(operation['control'], parameters)]
                if sum(storage_full)>0 and hour <= storage_full.index(1) and hour!=0:
                    continue
            
            current_mode = hours_ranked['mode'][rank]
            current_mode_print = f", in {str(current_mode)} mode." if current_mode!=np.nan else "."
            if PRINT: print(f"\nThe cheapest remaining hour before {next_unsatisfied}:00 is {hour}:00{current_mode_print}")

            # Quiet hours
            if (parameters['constraints']['quiet_hours']
                and hour in parameters['constraints']['quiet_hours_list'] 
                and operation['mode'][hour]=='Shed'): 
                if PRINT: 
                    print("\n[Constraint] Quiet hours")
                    print("[DONE] The selected hour is in quiet hours, it can not run at more than 'Shed' mode.")
                continue

            # Turn on the equipment with constraints in mind, eventually update maximum
            operation = turn_on(hour, operation, current_mode, hours_ranked, parameters, next_unsatisfied)

            # Update the status vector
            status = get_status(operation['control'], parameters)
            
            # Entire problem is solved
            if sum(status) == 0:
                if PRINT: 
                    print("\n" + "*"*30 + "\nProblem solved!\n" + "*"*30 + "\n")          
                    print(f"The total cost of the {parameters['horizon']} hours of operation is {sum(operation['cost'])}.\n")  
                if FINAL_PLOT: iteration_plot(operation, parameters)
                df_operation = pd.DataFrame(operation)
                return df_operation[['control','mode']] if parameters['control']['type']=='mode' else list(df_operation['control'])

            # The current next unsatisfied hour is now satisfied
            if next_unsatisfied != status.index(1):
                if PRINT: print(f"\nSatisfied hour {next_unsatisfied}:00, now next unsatisfied is {status.index(1)}:00.")
                next_unsatisfied = status.index(1)
                if INTERMEDIATE_PLOT: iteration_plot(operation, parameters)
                break
    

def get_hours_ranked(parameters):
    """
    Get hours ranked by cost per unit
    """

    # ---------------------------------------------------
    # Modes are involved in the ranking
    # ---------------------------------------------------

    if parameters['control']['type'] == 'mode':
        return parameters['control']['hours_ranked'].to_dict(orient='list')

    # ---------------------------------------------------
    # Modes are not involved in the ranking
    # ---------------------------------------------------

    # TODO: replace by actual costs per unit
    costs_pu = parameters['elec_costs']

    # Get rid of equal costs by adding a small random number
    needs_check = True
    while needs_check:
        needs_check = False
        for i in range(parameters['horizon']):
            for j in range(parameters['horizon']):
                if i!=j and costs_pu[i] == costs_pu[j]:
                    costs_pu[j] = round(costs_pu[j] + random.uniform(-0.001, 0.001),4)
                    needs_check = True

    # Rank hours by cost per unit and create dataframe
    hours_by_cost = [costs_pu.index(x) for x in sorted(costs_pu)]
    hours_ranked = {'hour':hours_by_cost, 'cost_pu':sorted(costs_pu), 'mode':[np.nan]*len(costs_pu)}
    
    return hours_ranked


def turn_on(hour, operation, hour_mode, hours_ranked, parameters, next_unsatisfied):
    """
    Turns on the system during the given hour, 
    while ensuring that constraints are respected
    """

    # Unpack and backup
    control = operation['control']
    control_max = operation['control_max']
    control_min = operation['control_min']
    mode = operation['mode']
    cost = operation['cost']
    control_backup = control.copy()

    # Apply the maximum control if there is no given mode
    if parameters['control']['type'] == 'range':
        control[hour] = control_max[hour]
        cost[hour] = control[hour] * parameters['elec_costs'][hour] 
        if parameters['hardware']['heatpump']:
            cost[hour] = cost[hour] / parameters['hardware']['COP'][hour]

    # Apply the given mode at that hour
    if parameters['control']['type'] == 'mode':
        mode[hour] = hour_mode
        control[hour], cost[hour] = get_mapping(hour_mode, hour, hours_ranked)

    # Check constraints that were marked as active
    for key, value in parameters['constraints'].items():
        if value:

            # ----------------------------
            # Maximum storage capacity
            # ----------------------------

            if key == "storage_capacity":
                if PRINT: print("\n[Constraint] Storage capacity")

                storage = [parameters['constraints']['initial_soc']] + [0]*parameters['horizon']
                max_storage = parameters['constraints']['max_storage']
                
                # Count the amount of storage excess
                storage_excess = round(max([x-max_storage if x-max_storage>0 else 0 for x in get_storage(control, parameters)]),1)

                # If there is no excess, stick to maximum
                if storage_excess == 0:
                    if PRINT: print(f"[DONE] The maximum control ({control[hour]}) can be used without exceeding the storage capacity.")

                else:
                    # See if you can reduce the power at that time
                    if control[hour] - storage_excess > control_min[hour] and control[hour] - storage_excess > control_backup[hour]:
                        if PRINT: print(f"[DONE] The control was lowered by {storage_excess} to avoid storage excess.")
                        # Set the control to the max power that does not exceed storage
                        control[hour] += -storage_excess
                        # Update the maximum
                        control_max[hour] += -storage_excess

                    # If not, keep at same power
                    else:
                        if PRINT: print(f"[DONE] The control could not be lowered by {storage_excess}, kept control at {control_backup[hour]}.")
                        control[hour] = control_backup[hour]

            # ----------------------------
            # Reaching cheaper hours
            # ----------------------------

            if key == "cheaper_hours":

                # Check the new current status
                status = get_status(control, parameters)

                # If the problem is not solved yet and the next unsatisfied hour has been satisfied
                if (sum(status) > 0 
                    and status.index(1) != next_unsatisfied 
                    and control[hour] > control_min[hour]):

                    costs = parameters['elec_costs']

                    # Check if a cheaper price has been passed while reaching the new next unsatisfied hour
                    if min(costs[next_unsatisfied:status.index(1)+1]) < costs[hour]:
                        if PRINT: print(f"\n[Constraint] Cheaper hour availability")

                        # Find the first hour in that section that is cheaper than the current hour
                        for price in costs[next_unsatisfied:status.index(1)+1]:
                            if price < costs[hour]:
                                hour_next_lower_price = costs.index(price)
                                break
                        
                        if PRINT: 
                            print(f"| The current control at {hour}:00 would satisfy all hours up to {status.index(1)}:00.")
                            print(f"| However, {hour_next_lower_price}:00 is cheaper than {hour}:00.")

                        # Initialize
                        storage = get_storage(control, parameters)   
                        lowest_storage_level = storage[hour_next_lower_price]
                        lowest_control = control[hour]
                                                
                        # Find the minimum power in that range that allows us to reach that cheaper hour
                        for c in range(int(control_min[hour]*10), int(control_max[hour]*10)+1):
                            
                            # Bring back the decimal
                            c /= 10
                            
                            # Define the new control sequence to test
                            test_control = control_backup.copy()
                            test_control[hour] = c

                            # Get the resulting status and storage sequence
                            test_status = get_status(test_control, parameters)
                            test_storage = get_storage(test_control, parameters)

                            # A lower control is valid only if the hour before the lower price hour is OK
                            if test_status[hour_next_lower_price-1]==0 and test_storage[hour_next_lower_price] < lowest_storage_level:
                                lowest_storage_level = test_storage[hour_next_lower_price]
                                lowest_control = c
                        
                        # Implement the solution
                        if PRINT: print(f"[DONE] Reduced the control to {lowest_control} to reach {hour_next_lower_price}:00 at the lowest cost.")
                        control[hour] = lowest_control
                    
                    else:
                        if PRINT: print("\n[Constraint] Cheaper hour availability\n[DONE] No cheaper hour was reached.")

    # Add the modifications at this hour to the solution dict
    operation['control'][hour] = control[hour]
    operation['control_max'][hour] = control_max[hour]
    operation['mode'][hour] = mode[hour]
    operation['cost'][hour] = cost[hour]

    return operation


def get_status(control, parameters):
    """
    Computes the status vector based on the given control sequence
    """

    # Extract from parameters
    N = parameters['horizon']
    load = parameters['load']['value']

    # When every hour must be satisfied
    if parameters['load']['type'] == 'hourly':

        storage = [parameters['constraints']['initial_soc']] + [0 for i in range(N)]
        status = [1]*N
    
        # If the control and storage can supply the load, then the status is OK
        for hour in range(N):
            if storage[hour] + control[hour] >= load[hour]:
                storage[hour+1] = storage[hour] + control[hour] - load[hour]
                status[hour] = 0

    # When a full day must be satisfied
    elif parameters['load']['type'] == 'daily':
        
        # Status is OK when daily load is satisfied
        if sum(control) >= parameters['load']['value']:
            status = [0]*N
        else:
            status = [0]*(N-1) + [1]

    return status


def get_storage(control, parameters):
    """
    Computes the current storage levels for a given control sequence
    """

    storage = [parameters['constraints']['initial_soc']] + [0]*parameters['horizon']
    load = parameters['load']['value']

    for hour in range(parameters['horizon']):
        if storage[hour] + control[hour] >= load[hour]:
            storage[hour+1] = storage[hour] + control[hour] - load[hour]
    
    return storage


def iteration_plot(operation, parameters):
    """
    Plots the current iteration
    """

    # Extract
    N = parameters['horizon']
    control = operation['control']
    costs_pu = parameters['elec_costs']

    fig, ax = plt.subplots(1,1, figsize=(13,4))
    ax2 = ax.twinx()

    # Plot the controls and prices
    controls_plot = control + [control[-1]]
    ax.step(range(N+1), controls_plot, where='post', color='blue', alpha=0.5, label='Heat pump')
    costs_plot = costs_pu + [costs_pu[-1]]
    ax2.step(range(N+1), costs_plot, where='post', color='gray', alpha=0.6, label='Electricity price')

    # Plot hourly loads
    if parameters['load']['type']=='hourly':
        loads_plot = parameters['load']['value'] + [parameters['load']['value'][-1]]
        ax.step(range(N+1), loads_plot, where='post', color='red', alpha=0.4, label='Load')

    # Plot the storage levels
    if parameters['constraints']['storage_capacity']:
        ax.plot(range(N+1), get_storage(control, parameters), alpha=0.6, color='orange', label='Storage')
        ax.plot(range(N+1), [parameters['constraints']['max_storage']]*(N+1), alpha=0.5, linestyle='dotted', color='orange', label='Maximum storage')

    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Energy [kWh]")
    ax2.set_ylabel("Cost [cts/kWh]")

    ax.set_xticks(range(N+1))
    
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()


def check_parameters(parameters):
    """
    Checks that the entered parameters are coherent
    """

    if len(parameters['elec_costs']) != parameters['horizon']:
        raise ValueError("The length of the electricity price forecast must match the specified horizon.")

    if (parameters['load']['type'] == 'hourly' 
        and len(parameters['load']['value']) != parameters['horizon']):
            raise ValueError("The length of the load forecast must match the specified horizon.")
    
    if parameters['control']['type'] == 'range':
        if (len(parameters['control']['max']) != parameters['horizon']
            or len(parameters['control']['min']) != parameters['horizon']):
            raise ValueError("The length of the control ranges must match the specified horizon.")