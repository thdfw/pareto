import numpy as np
import pandas as pd

PRINT = False

def get_commands(Q_HP, load):

    # ---------------------------------------
    # Get d_HP, mode, load
    # ---------------------------------------  

    # **** Q_HP = 0 ****
    # Mode 3 (PCM->Load) whenever the HP is off
    delta_HP = [0 if q==0 else 1 for q in Q_HP for _ in range(60)]
    modes = [3 if q==0 else np.nan for q in Q_HP for _ in range(60)]
    loads = [load[i] if Q_HP[i]==0 else np.nan for i in range(len(Q_HP)) for _ in range(60)]

    # **** Q_HP > Q_load ****
    # Mode 1 (HP->Load) until the load is satisfied
    time_mode1 = [0 if Q_HP[i]==0 else round(load[i]/Q_HP[i]*60) for i in range(len(Q_HP))]
    if PRINT: print(f"Mode 1 for {time_mode1} minutes.")
    for hour in range(len(Q_HP)):
        for minute in range(time_mode1[hour]):
            modes[hour*60+minute] = 1
            loads[hour*60+minute] = Q_HP[hour]
    # Mode 2 (HP->PCM) after that
    time_mode2 = [0 if Q_HP[i]==0 else round(60-load[i]/Q_HP[i]*60) for i in range(len(Q_HP))]
    if PRINT: print(f"Mode 2 for {time_mode2} minutes.")
    for hour in range(len(Q_HP)):
        for minute in range(time_mode2[hour]):
            modes[hour*60+time_mode1[hour]+minute] = 2
            loads[hour*60+time_mode1[hour]+minute] = 0

    # ---------------------------------------
    # Get T_sup_HP
    # --------------------------------------- 

    T_ret_PCM = 58 
    T_HP_min = 55
    T_HP_max = 65
    
    Q_HP_by_min = [x for x in Q_HP for _ in range(60)]
    T_sup_HP = [T_HP_min if q==0 else np.nan for q in Q_HP for _ in range(60)]

    count_min_mode2 = 0

    for min in range(len(T_sup_HP)):
        # Assume a constant temperature drop at the load for now
        if modes[min]==1 and min%60==0:
            T_sup_HP[min] = 60
        elif modes[min]==1 and min>0:
            T_sup_HP[min] = round(Q_HP_by_min[min]*1000/0.29/4187 + (T_sup_HP[min-1] - 10),1)
        # Assume a constant return water temperature from the PCM
        elif modes[min]==2:
            count_min_mode2 += 1
            if count_min_mode2==1:
                T_sup_HP[min] = round(Q_HP_by_min[min]*1000/0.29/4187 + T_sup_HP[min-1]-10,1)
            elif count_min_mode2<5:
                T_sup_HP[min] = round(Q_HP_by_min[min]*1000/0.29/4187 + 50 + 2.33*count_min_mode2,1)
            else:
                T_sup_HP[min] = round(Q_HP_by_min[min]*1000/0.29/4187 + T_ret_PCM,1)
        
        if T_sup_HP[min] > T_HP_max:
            T_sup_HP[min] = T_HP_max
        if T_sup_HP[min] < T_HP_min and T_sup_HP[min]!=0:
            T_sup_HP[min] = T_HP_min

    # ---------------------------------------
    # Print results
    # --------------------------------------- 

    for hour in range(len(Q_HP)):
        if PRINT and hour==0:
            print(f"\nHour {hour}")
            print(f"OnOff: {delta_HP[hour*60:hour*60+60]}")
            print(f"Mode: {modes[hour*60:hour*60+60]}")    
            print(f"Load: {[round(x,2) for x in loads[hour*60:hour*60+60]]}")
            print(f"Tsupply: {T_sup_HP[hour*60:hour*60+60]}")

    commands = {
        'mode': modes,
        'delta_HP': delta_HP,
        'load': loads,
        'T_sup_HP': T_sup_HP,
    }

    return commands


''' 
# EXAMPLE
Q_HP = [12.0, 0, 12.0, 12.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.2, 12.0, 12.0, 12.0, 12.0, 0.0, 9.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0]   
load = [5.91, 5.77, 5.67, 5.77, 5.71, 6.06, 6.34, 6.34, 6.01, 5.77, 5.05, 5.05, 4.91, 4.91, 4.91, 4.91, 5.05, 5.1, 4.91, 4.91, 4.91, 4.91, 4.98, 4.91]
commands = get_commands(Q_HP, load)
print(pd.DataFrame(commands))
'''