import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import casadi
import os

PRINT = False

# --------------------------------
# --------------------------------
# Manufacturer data
# --------------------------------
# --------------------------------

Q_max_table = [[0,-25,-20,-15,-7,-4,-2,2,7,10,15,18,20,35],
      [30,8750,10130,11500,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000],
      [35,8500,10000,11500,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000],
      [40,8250,9880,11500,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000],
      [45,8000,9750,11500,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000],
      [50,9630,9630,11500,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000],
      [55,11500,11500,11500,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000],
      [60,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000],
      [65,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000,12000]]

COP_table = [[0,-25,-20,-15,-7,-4,-2,2,7,10,15,18,20,35],
      [30,2.13,2.34,2.55,3.15,3.36,3.47,3.69,4.93,5.22,5.99,6.29,6.49,7.98],
      [35,1.85,2.13,2.40,3.00,3.17,3.28,3.50,4.60,4.87,5.56,5.84,6.02,7.41],
      [40,1.58,1.91,2.25,2.85,2.97,3.09,3.31,4.27,4.51,5.13,5.39,5.56,6.84],
      [45,1.30,1.70,2.10,2.70,2.78,2.90,3.12,3.93,4.16,4.71,4.94,5.10,6.28],
      [50,1.49,1.49,1.95,2.55,2.59,2.71,2.93,3.60,3.81,4.28,4.49,4.64,5.71],
      [55,1.80,1.80,1.80,2.40,2.39,2.53,2.73,2.80,3.46,3.85,4.05,4.17,5.14],
      [60,2.25,2.25,2.25,2.25,2.20,2.34,2.54,2.60,3.10,3.43,3.60,3.71,4.57],
      [65,2.05,2.05,2.05,2.05,2.05,2.15,2.35,2.60,2.75,3.00,3.15,3.25,4.00]]

# Mass flow rate depends on LWT
def m_HP(T):
    if   T<=45: m = 34.5/60
    elif T<=55: m = 21.6/60
    elif T<=65: m = 17.3/60
    return round(m,2)
    
# Maximum heating power
Q_max = pd.DataFrame(Q_max_table[1:], columns=Q_max_table[0]).T
Q_max.columns = Q_max.iloc[0]
Q_max = Q_max[1:]

# Minimum heating power (assumed to be 30% of the maximum)
Q_min = Q_max * 0.3
Q_min = Q_min.round(1)

# COP
COP = pd.DataFrame(COP_table[1:], columns=COP_table[0]).T
COP.columns = COP.iloc[0]
COP = COP[1:]

# --------------------------------
# --------------------------------
# Polynomial fits of the tables
# --------------------------------
# --------------------------------
def get_range_and_COP(LWT, T_OA):

    # Get the Q_max and COP polyfits
    Q_max_polyfit = np.polyfit(list(Q_max.index), list(Q_max[LWT]), 3)
    Q_min_polyfit = np.polyfit(list(Q_min.index), list(Q_min[LWT]), 3)
    COP_polyfit = np.polyfit(list(COP.index), list(COP[LWT]), 3)

    # Get value for a given temperature forecast
    Q_HP_max = np.polyval(Q_max_polyfit, T_OA)
    Q_HP_min = np.polyval(Q_min_polyfit, T_OA)
    COPs = np.polyval(COP_polyfit, T_OA)

    return Q_HP_max, Q_HP_min, COPs
    
# --------------------------------
# --------------------------------
# Get price (kWh_th) and heating range for a given hour
# --------------------------------
# --------------------------------

# Get the LWT options available at a given T_OA and T_HP_in
def get_LWT_options(T_OA, elec, T_HP_in, T_HP_out_min, PRINT):
        
    # Range (min,max) of acceptable Q_HP for each LWT option at the given T_OA
    LWT_options = [list[0] for list in Q_max_table][1:]
    
    # Get the corresponding Q_min and Q_max depending on T_OA
    Q_min_by_LWT, Q_max_by_LWT = [], []
    for LWT in LWT_options:
        Q_HP_max, Q_HP_min, _ = get_range_and_COP(LWT, T_OA)
        Q_max_by_LWT.append(Q_HP_max)
        Q_min_by_LWT.append(Q_HP_min)
    
    # Find the LWT(s) that are attainable within this range of Q_HP
    available_LWT = []
    for i in range(len(LWT_options)):
        # Make sure the LWT is above the given minimum
        if LWT_options[i] < T_HP_out_min: continue
        # The heat needed to attain that LWT
        Q_HP = round(m_HP(LWT_options[i])*4187*(LWT_options[i]-T_HP_in),1)
        # If it can be attained
        if Q_HP <= Q_max_by_LWT[i] and Q_HP >= Q_min_by_LWT[i]:
            if PRINT: print(f"[OK] LWT = {LWT_options[i]}°C, m_HP = {m_HP(LWT_options[i])} kg/s \
            would use Q_HP = {round(Q_HP/1000,1)} kW")
            available_LWT.append(LWT_options[i])
        # If it can not be attained
        else:
            if PRINT: print(f"[--] LWT = {LWT_options[i]}°C, m_HP = {m_HP(LWT_options[i])} kg/s \
            would use Q_HP = {round(Q_HP/1000,1)} kW")
    
    # The corresponding prices and Q_HP ranges
    for LWT in available_LWT:
        Q_HP = m_HP(LWT)*4187*(LWT-T_HP_in)/1000
        Q_max_LWT_TOA, _, COP_LWT_TOA = get_range_and_COP(LWT, T_OA)
        W_HP = Q_HP/COP_LWT_TOA
        
        if PRINT:
            print(f"\n{LWT}°C water is possible:")
            print(f"Requires {round(Q_HP,1)} kW_th => {round(W_HP,1)} kW_elec at {round(W_HP*elec,1)} cts/kWh => [{round(W_HP*elec/Q_HP,2)} cts/kWh_th]")
            print(f"The Q_HP range for this LWT is => [{round(Q_HP,1)}, {round(Q_max_LWT_TOA/1000)}] kW")

    # There is generally only one LWT option. Save the range and cost for the given hour.
    Q_HP_min = round(Q_HP,3)
    Q_HP_max, _, __ = get_range_and_COP(LWT, T_OA)
    Q_HP_max = round(Q_HP_max/1000,3)
    cost_th = round(W_HP*elec/Q_HP,3)
    
    return Q_HP_min, Q_HP_max, cost_th, m_HP(LWT)
    
# --------------------------------
# --------------------------------
# Get price (kWh_th) and heating range for all hours
# --------------------------------
# --------------------------------

def get_costs_and_ranges(price_forecast, T_OA_list, T_HP_in, T_HP_out_min, num_hours, iter):

    # --------------------------------
    # Price forecasts and parameters
    # --------------------------------

    # Electricity prices [cts/kWh] option 1 (GridWorks)
    if price_forecast=="GridWorks":
        c_el = [6.36, 6.34, 6.34, 6.37, 6.41, 6.46, 6.95, 41.51,
        41.16, 41.07, 41.06, 41.08, 7.16, 7.18, 7.18, 7.16, 41.2, 41.64,
        41.43, 41.51, 6.84, 6.65, 6.46, 6.4]

    # Electricity prices [cts/kWh] option 2 (progressive)
    elif price_forecast=="Progressive":
        c_el = [18.97, 18.92, 18.21, 16.58, 16.27, 15.49, 14.64,
        18.93, 45.56, 26.42, 18.0, 17.17, 16.19, 30.74, 31.17, 16.18,
        17.11, 20.24, 24.94, 24.69, 26.48, 30.15, 23.14, 24.11]

    # Electricity prices [cts/kWh] option 3 (Peter)
    elif price_forecast=="Peter":
        c_el = [0.07919, 0.066283, 0.063061, 0.067943, 0.080084, 0.115845,
        0.193755, 0.215921, 0.110822, 0.044927, 0.01521, 0.00742,
        0.004151, 0.007117, 0.009745, 0.02452, 0.037877, 0.09556,
        0.205067, 0.282588, 0.234866, 0.184225, 0.132268, 0.101679]
        c_el = [x*100 for x in c_el]
    
    elif price_forecast=="CFH":
        c_el = [0.0359, 0.0206, 0.0106, 0.0192, 0.0309, 0.0612, 0.0925, 0.1244, 0.1667, 0.2148, 0.3563,
        0.4893, 0.7098, 0.7882, 0.5586, 0.3326, 0.2152, 0.1487, 0.0864, 0.0587, 0.0385, 0.0246, 0.0165, 0.0215]
        c_el = [x*100 for x in c_el]
        
    elif price_forecast=="year":
        df = pd.read_excel(os.getcwd()+'/data/gridworks_yearly_data.xlsx', header=3, index_col = 0)
        df.index = pd.to_datetime(df.index)
        df.index.name = None
        df['c_el'] = df['Rt Energy Price ($/MWh)'] + df['Dist Price ($/MWh)']
        c_el = df['c_el'].tolist()
        c_el += list(df['c_el'][:24])

    # If it doesn't match assume the prices were given directly
    else:
        c_el = price_forecast
        
    # Get the future prices at the current iteration time
    c_el = c_el * 2
    c_el = c_el[iter%24:]
    c_el = c_el[:num_hours]
    if PRINT: print(f"Cost forecast:\n{c_el}")
        
    # Water returning from the PCM (°C)
    if PRINT: print(f"Assuming water going to the HP at {T_HP_in}°C")
    
    # Obtain the costs and available heating ranges for every hour
    Q_HP_min_list, Q_HP_max_list, cost_th_list, m_HP_list = [], [], [], []

    for i in range(num_hours):

        if PRINT:
            print("\n----------------------------------------")
            print(f"Hour {i+1} ({T_OA_list[i]}°C outside, {round(c_el[i],2)} cts/kWh)")
            print("----------------------------------------\n")
        
        # Q_HP_min, Q_HP_max, cost_th, m_HP = get_LWT_options(T_OA_list[i], c_el[i], T_HP_in, T_HP_out_min, PRINT)
    
        # Append the values in lists
        Q_HP_min = 6
        Q_HP_max = 12
        cost_th = 4/12 * c_el[i]
        m_HP = 17.3/60
        Q_HP_min_list.append(round(Q_HP_min,1))
        Q_HP_max_list.append(round(Q_HP_max,1))
        cost_th_list.append(cost_th)
        m_HP_list.append(m_HP)
        
    return Q_HP_min_list, Q_HP_max_list, cost_th_list, m_HP_list

# --------------------------------
# --------------------------------
# Pareto algorithm
# --------------------------------
# --------------------------------

def get_pareto(load, price_forecast, T_OA_list, T_HP_in, T_HP_out_min, max_storage, PRINT, PLOT, num_hours, CIs, iter, SoC_current):

    # Get heating ranges and costs for each hour
    Q_HP_min_list, Q_HP_max_list, cost_th_list, m_HP_list = get_costs_and_ranges(price_forecast, T_OA_list, T_HP_in, T_HP_out_min, num_hours, iter)

    if PRINT:
        print(f"Q_HP_min_list = {Q_HP_min_list}")
        print(f"Q_HP_max_list = {Q_HP_max_list}")
        print(f"cost_th_list = {cost_th_list}")

    # Horizon
    N = len(cost_th_list)

    # Treat for duplicates
    for i in range(N):
        for j in range(N):
            if i!=j and cost_th_list[i] == cost_th_list[j]:
                cost_th_list[j] = cost_th_list[j] + random.uniform(-0.5, 0.5)

    # Ranking the hourly costs ($/kWh_th)
    ranking = [sorted(cost_th_list).index(x) for x in cost_th_list]
    if PRINT: print(f"Ranking hours by $/kWh_th:\n{ranking}\n")

    # Initialize
    storage = [SoC_current] + [0 for i in range(N)]
    Q_HP = [0 for i in range(N)]
    problem_solved = False
    ok = [1]*N
    last_not_ok = N-2
    testing = storage.copy()
    Q_HP_discovered_max = Q_HP_max_list.copy()
    
    for j in range(N):
        if Q_HP[j]-load[j]+storage[j] >= 0:
            storage[j+1] = Q_HP[j]-load[j]+storage[j]
            ok[j] = 0

    #------------------------------------------------------
    # Starting at the first not ok hour
    # Turn on the HP before or during that hour
    # From the lowest to the highest price hour
    # Until that hour is ok
    # Move to next not ok hour and repeat
    #------------------------------------------------------

    while sum(ok) != 0:
        
        first_not_ok = ok.index(1)

        # Backups
        storage_backup = storage.copy()
        Q_HP_backup = Q_HP.copy()
        ok_backup = ok.copy()
        
        if PRINT:
            print("---------------------------------------")
            print(f"The first unsatisfied hour: {first_not_ok}:00")
            print("---------------------------------------")

        # For all hours by ranking
        for i in range(N):
        
            # Skip all hours after the first unsatisfied hour
            if ranking.index(i) > first_not_ok: continue

            # Skip all hours that are already turned on
            #if Q_HP[ranking.index(i)] != 0: continue
            if Q_HP[ranking.index(i)] == Q_HP_max_list[ranking.index(i)]: continue

            if PRINT:
                print(f"{ranking.index(i)}:00, is the cheapest remaining hour before {first_not_ok}:00")
            
            #------------------------------------------------------
            # Use the max Q_HP you can in the cheapest remaining hour
            #------------------------------------------------------

            # Try the maximum Q_HP
            Q_HP[ranking.index(i)] = Q_HP_max_list[ranking.index(i)]
            
            # Check if this violates the max storage constraint
            total_violation = 0
            testing = storage.copy()
            for j in range(N):
                if Q_HP[j]-load[j]+testing[j] >= 0:
                    testing[j+1] = Q_HP[j]-load[j]+testing[j]
                    if testing[j+1] > max_storage:
                        total_violation += testing[j+1]-max_storage
            if total_violation != 0: total_violation = round(total_violation[0],1)

            # Max storage is not violated: use max power
            if total_violation == 0:
                if PRINT: print(f"The maximum Q_HP can be used at this time.\n")
                for j in range(N):
                    if Q_HP[j]-load[j]+storage[j] >= 0:
                        storage[j+1] = Q_HP[j]-load[j]+storage[j]
                        ok[j] = 0
        
            # Max storage is violated: use exact power or turn off
            else:
                if PRINT: print(f"Need to use {total_violation} less kW than the maximum.")
                
                # See if you can reduce the Q_HP at that time
                # Only if Q_HP is currently 0
                if Q_HP_max_list[ranking.index(i)] - total_violation > Q_HP_min_list[ranking.index(i)]\
                and Q_HP_backup[ranking.index(i)]==0:
                    if PRINT: print(f"Feasible, reduced the HP power.\n")
                    Q_HP[ranking.index(i)] = Q_HP_max_list[ranking.index(i)] - total_violation
                    Q_HP_discovered_max[ranking.index(i)] = Q_HP_max_list[ranking.index(i)] - total_violation
                    for j in range(N):
                        if Q_HP[j]-load[j]+storage[j] >= 0:
                            storage[j+1] = Q_HP[j]-load[j]+storage[j]
                            ok[j] = 0
                # If not, turn off the HP at that time
                else:
                    if PRINT: print(f"Infeasible, keeped HP at current state {Q_HP_backup[ranking.index(i)]}.\n")
                    Q_HP[ranking.index(i)] = Q_HP_backup[ranking.index(i)]
                    for j in range(N):
                        if Q_HP[j]-load[j]+storage[j] >= 0:
                            storage[j+1] = Q_HP[j]-load[j]+storage[j]
                            ok[j] = 0

            #------------------------------------------------------
            # Tried to go as far as possible
            # Were there cheaper prices between the current hour and how far we went?
            # If so, try to reach these cheaper prices
            #------------------------------------------------------

            if sum(ok) > 0 and ok.index(1) != first_not_ok:
            
                # first_minimum available
                for price in cost_th_list[first_not_ok:ok.index(1)+1]:
                    if price < cost_th_list[ranking.index(i)]:
                        minimum_available = price
                        hour_min_available = first_not_ok + cost_th_list[first_not_ok:ok.index(1)+1].index(price)
                        break
                    else:
                        minimum_available = cost_th_list[ranking.index(i)]
                
                #minimum_available = np.min(cost_th_list[first_not_ok:ok.index(1)+1])
                #hour_min_available = first_not_ok + cost_th_list[first_not_ok:ok.index(1)+1].index(minimum_available)
                current_price = cost_th_list[ranking.index(i)]

                # A cheaper hour than now is attained
                if minimum_available < cost_th_list[ranking.index(i)]:
                    
                    # Try to use a lower Q_HP to reach that point
                    if Q_HP[ranking.index(i)] > Q_HP_min_list[ranking.index(i)]\
                    and Q_HP[hour_min_available] < Q_HP_max_list[hour_min_available] :
                        
                        if PRINT:
                            print(f"Everything was ok until {first_not_ok}, now until {ok.index(1)}")
                            print(f"A cheaper price (now: {cost_th_list[ranking.index(i)]}) was reached at hour {hour_min_available}: {minimum_available}.")
                            print(f"A lower Q_HP was possible!")
                            print(f"Currently Q_HP={Q_HP_backup[ranking.index(i)]}")
                            
                        # Find the right Q_HP to use
                        q_mini = int(Q_HP_min_list[ranking.index(i)] * 10) if int(Q_HP_backup[ranking.index(i)])==0 else int(Q_HP[ranking.index(i)]*10)
                        q_maxi = int(Q_HP_discovered_max[ranking.index(i)] * 10)
                        min_so_far = storage[hour_min_available]
                        best_q = Q_HP[ranking.index(i)]

                        # Find the minimum Q_HP in that range that satisfied storage[min hour] > 0
                        if PRINT: print(f"Looking into range: ({q_mini},{q_maxi})")
                        for fake_q in range(q_mini, q_maxi+1):
                            
                            real_q = fake_q/10
                            
                            test_storage = storage_backup.copy()
                            test_Q_HP = Q_HP_backup.copy()
                            test_ok = ok_backup.copy()

                            test_Q_HP[ranking.index(i)] = real_q
                            
                            for j in range(N):
                                if test_Q_HP[j]-load[j]+test_storage[j] >= 0:
                                    test_storage[j+1] = test_Q_HP[j]-load[j]+test_storage[j]
                                    test_ok[j] = 0

                            # Just before the min hour need to be ok
                            if hour_min_available>0 and test_ok[hour_min_available-1] == 0:
                                # Check the storage at min hour
                                if test_storage[hour_min_available] < min_so_far:
                                    min_so_far = test_storage[hour_min_available]
                                    best_q = real_q

                        # Implement it
                        if PRINT: print(f"A better Q_HP was {best_q}")
                        storage = storage_backup.copy()
                        Q_HP = Q_HP_backup.copy()
                        ok = ok_backup.copy()

                        Q_HP[ranking.index(i)] = best_q
                        for j in range(N):
                            if Q_HP[j]-load[j]+storage[j] >= 0:
                                storage[j+1] = Q_HP[j]-load[j]+storage[j]
                                ok[j] = 0
                        if PRINT: print("")

            #------------------------------------------------------
            # Plot the current iteration
            #------------------------------------------------------

            if PRINT and sum(ok) > 0 and ok.index(1) != first_not_ok and PLOT:

                # Duplicate the last element of the hourly data for the plot
                cost_th_list2 = cost_th_list + [cost_th_list[-1]]
                Q_HP2 = [round(x,3) for x in Q_HP + [Q_HP[-1]]]
                load2 = load + [load[-1]]

                # Plot the current state of the system
                fig, ax = plt.subplots(1,1, figsize=(13,4))
                ax2 = ax.twinx()
                ax2.step(range(N+1), cost_th_list2, where='post', color='gray', alpha=0.6, label='Cost per kWh_th')
                ax.step(range(N+1), load2, where='post', color='red', alpha=0.4, label='Load')
                ax.step(range(N+1), Q_HP2, where='post', color='blue', alpha=0.5, label='HP')
                ax.plot(range(N+1), [max_storage]*(N+1), alpha=0.2, linestyle='dotted', color='gray')
                print(storage)
                #storage = [round(x,2) for x in storage]
                #ax.plot(storage, color='orange', alpha=0.6, label='Storage')
                ax.set_ylim([0,35])
                ax.set_xticks(range(N+1))
                ax.set_xlabel("Time [hours]")
                ax.set_ylabel("Heat [kWh_th]")
                ax2.set_ylabel("Price [cts/kWh_th]")
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2)
                plt.show()

            #------------------------------------------------------
            # Check up to when it is ok
            #------------------------------------------------------
            
            # If the first_not_ok hour is satisfied, break
            if sum(ok) > 0:
                if ok.index(1) != first_not_ok:
                    if PRINT: print(f"Success! Everything until {ok.index(1)}:00 is satisfied.\n")
                    break

            # If the problem is solved, break
            if sum(ok) == 0:
                problem_solved = True
                if PRINT: print("Problem solved!\n")
                break

    #------------------------------------------------------
    # Calculate the total cost and energy
    #------------------------------------------------------

    total_cost = 0
    for i in range(N):
        total_cost += Q_HP[i]*cost_th_list[i]
    total_cost = round(total_cost/100,2)
    total_energy = round(sum(Q_HP),1)

    #------------------------------------------------------
    # Run a check
    #------------------------------------------------------
    for j in range(N):
        if Q_HP[j] < 0:
            print("MAYDAY")
            raise ValueError("MAYDAY")
        if Q_HP[j] != 0 and (Q_HP[j]<Q_HP_min_list[j] or Q_HP[j]>Q_HP_max_list[j]):
            print("MAYDAY")
            raise ValueError("MAYDAY")
        if Q_HP[j]-load[j]+storage[j] < 0:
            print("MAYDAY")
            raise ValueError("MAYDAY")
        if storage[j] > max_storage+0.05:
            print(f"Failed at {j}: {storage[j]}")
            print("MAYDAY")
            raise ValueError("MAYDAY")
            
    #------------------------------------------------------
    # Plot
    #------------------------------------------------------
    PLOT = True
    if PLOT:
        # Duplicate the last element of the hourly data for the plot
        N = len(cost_th_list)
        cost_th_list2 = cost_th_list + [cost_th_list[-1]]
        Q_HP_plot = [round(x,3) for x in Q_HP + [Q_HP[-1]]]
        load2 = load + [load[-1]]
        
        # Get the storage
        flat_list = []
        for item in storage:
            if isinstance(item, np.ndarray) and item.size == 1:
                flat_list.append(item.item())
            else:
                flat_list.append(item)
        storage = flat_list

        # Plot the state of the system
        fig, ax = plt.subplots(1,1, figsize=(13,4))
        plt.title(f"Cost: {total_cost}$, Supplied: {total_energy} kWh_th \n=> {round(100*total_cost/total_energy,2)} cts/kWh_th")
        ax2 = ax.twinx()
        ax2.step(range(N+1), cost_th_list2, where='post', color='gray', alpha=0.6, label='Cost per kWh_th')
        ax.step(range(N+1), load2, where='post', color='red', alpha=0.4, label='Load')
        ax.step(range(N+1), Q_HP_plot, where='post', color='blue', alpha=0.5, label='HP')
        ax.plot(storage, color='orange', alpha=0.6, label='Storage')
        ax.plot(range(N+1), [max_storage]*(N+1), alpha=0.2, linestyle='dotted', color='gray')
        ax.set_ylim([0,35])
        ax.set_xticks(range(N+1))
        ax.set_xlabel("Time [hours]")
        ax.set_ylabel("Heat [kWh_th]")
        ax2.set_ylabel("Price [cts/kWh_th]")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)
        
        # Plot confidence intreval
        load = [x[0] for x in load2]
        CIs = CIs + [CIs[-1]]
        lower_bounds = [load[i] - CIs[i] for i in range(len(CIs))]
        upper_bounds = [load[i] + CIs[i] for i in range(len(CIs))]
        ax.fill_between(range(len(CIs)), lower_bounds, upper_bounds, color='red', alpha=0.05, label='90% confidence interval')

        plt.show()

    return Q_HP, m_HP_list