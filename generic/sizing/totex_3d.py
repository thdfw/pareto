import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import casadi
import os
import sys
from matplotlib.ticker import FuncFormatter

# Load [kWh]
load = [5.117, 5.117, 5.117, 5.117, 5.117, 5.117, 4.652, 4.652, 3.722, 3.722, 3.722, 3.722, 
        3.722, 3.257, 3.257, 3.257, 3.257, 3.257, 4.187, 4.187, 4.652, 4.652, 5.117, 5.117]

# To try variations
# load = [x/2 for x in load]

# Outside air temperature (°C)
T_OA = [12]*24

# Electricity prices [cts/kWh]
price_type = "gridworks"

if price_type == "gridworks":
    c_el = [6.36, 6.34, 6.34, 6.37, 6.41, 6.46, 6.95, 41.51,
            41.16, 41.07, 41.06, 41.08, 7.16, 7.18, 7.18, 7.16, 41.2, 41.64,
            41.43, 41.51, 6.84, 6.65, 6.46, 6.4]

elif price_type == "progressive":
    c_el = [18.97, 18.92, 18.21, 16.58, 16.27, 15.49, 14.64,
            18.93, 45.56, 26.42, 18.0, 17.17, 16.19, 30.74, 31.17, 16.18,
            17.11, 20.24, 24.94, 24.69, 26.48, 30.15, 23.14, 24.11]

elif price_type == "peter":
    c_el = [0.07919, 0.066283, 0.063061, 0.067943, 0.080084, 0.115845, 
                0.193755, 0.215921, 0.110822, 0.044927, 0.01521, 0.00742, 
                0.004151, 0.007117, 0.009745, 0.02452, 0.037877, 0.09556, 
                0.205067, 0.282588, 0.234866, 0.184225, 0.132268, 0.101679]
    c_el = [x*100 for x in c_el]

B0_Q, B1_Q = -68851.589, 313.3151
B0_C, B1_C = 2.695868, -0.008533

def Q_HP_max(T_OA):
    T_OA += 273
    return round((B0_Q + B1_Q*T_OA)/1000,2) if T_OA<273-7 else 12
    
def COP1(T_OA):
    T_OA += 273
    return round(B0_C + B1_C*T_OA,2)

COP1_list = [COP1(temp) for temp in T_OA]

def get_opti(N, c_el, load, max_storage, storage_initial, Q_HP_min_list, Q_HP_max_list):

    # Initialize    
    #opti = casadi.Opti()
    opti = casadi.Opti('conic')
 
    # -----------------------------
    # Variables and solver
    # -----------------------------
    
    storage = opti.variable(1,N+1)  # state
    Q_HP = opti.variable(1,N)       # input
    delta_HP = opti.variable(1,N)  # input
    Q_HP_onoff = opti.variable(1,N) # input (derived)
    
    # delta_HP is a discrete variable (binary)
    discrete_var = [0]*(N+1) + [0]*N + [1]*N + [0]*N

    # Solver
    opti.solver('gurobi', {'discrete':discrete_var, 'gurobi.OutputFlag':0})
    #opti.solver('bonmin', {'discrete': discrete_var, 'bonmin.tol': 1e-4, 'bonmin.print_level': 0, 'print_time': 0})

    # -----------------------------
    # Constraints
    # -----------------------------
    
    # Initial storage level
    opti.subject_to(storage[0] == storage_initial)

    # Constraints at every time step
    for t in range(N+1):

        # Bounds on storage
        opti.subject_to(storage[t] >= min_storage)
        opti.subject_to(storage[t] <= max_storage)

        if t < N:
            
            # System dynamics
            opti.subject_to(storage[t+1] == storage[t] + Q_HP_onoff[t] - load[t])
    
            # Bounds on delta_HP
            opti.subject_to(delta_HP[t] >= 0)
            opti.subject_to(delta_HP[t] <= 1)
        
            # Bounds on Q_HP
            opti.subject_to(Q_HP[t] <= Q_HP_max_list[t])    
            opti.subject_to(Q_HP[t] >= Q_HP_min_list[t]*delta_HP[t])
        
            # Bilinear to linear
            opti.subject_to(Q_HP_onoff[t] <= Q_HP_max_list[t]*delta_HP[t])
            opti.subject_to(Q_HP_onoff[t] >= Q_HP_min_list[t]*delta_HP[t])
            opti.subject_to(Q_HP_onoff[t] <= Q_HP[t] + Q_HP_min_list[t]*(delta_HP[t]-1))
            opti.subject_to(Q_HP_onoff[t] >= Q_HP[t] + Q_HP_max_list[t]*(delta_HP[t]-1))
    
    # -----------------------------
    # Objective
    # -----------------------------
    
    obj = sum(Q_HP_onoff[t]*c_el[t]*COP1_list[t] for t in range(N))
    opti.minimize(obj)

    # -----------------------------
    # Solve and get optimal values
    # -----------------------------

    sys.stdout = open(os.devnull, 'w')
    sol = opti.solve()
    sys.stdout = sys.__stdout__

    Q_opt = sol.value(Q_HP_onoff)
    stor_opt = sol.value(storage)
    HP_on_off_opt = sol.value(delta_HP)
    obj_opt = round(sol.value(obj)/100,2)

    return Q_opt, stor_opt, HP_on_off_opt, obj_opt

# Horizon (hours)
N = 24

# The storage capacity (kWh)
min_storage = 0
storage_initial = 0

# Lifetime of equipement
lifetime_HP = 10
lifetime_TES = 10

def capex_HP(max_Q_HP):
    A = [5,8,12,16]
    C = [1984,2235,3257,3616]
    C = [x/365/lifetime_HP for x in C]
    C_polyfit = np.polyfit(A, C, 2)
    return(np.polyval(C_polyfit, max_Q_HP))

def capex_TES(max_storage):
    A = [6, 9, 12]
    C = [2416, 2857, 3297]
    C = [x/365/lifetime_TES for x in C]
    C_polyfit = np.polyfit(A, C, 2)
    return(np.polyval(C_polyfit, max_storage))

def get_TOTEX(max_storage, max_Q_HP):#, load):
    
    # The heat pump capacity (kW)
    min_Q_HP = 0.1*max_Q_HP
    Q_HP_max_list = [max_Q_HP]*N
    Q_HP_min_list = [min_Q_HP]*N
    
    # Get the optimal cost
    try:
        _, __, ___, obj_opt = get_opti(N, c_el, load, max_storage, storage_initial, Q_HP_min_list, Q_HP_max_list)

        #print(f"Max stor: {max_storage} kWh, HP: {max_Q_HP} kWh")
        #print(f"OPEX: {round(obj_opt,2)}, CAPEX HP: {capex_HP(max_Q_HP)}, CAPEX TES: {capex_TES(max_storage)}")
        ##obj_opt += capex_HP(max_Q_HP)
        ##obj_opt += capex_TES(max_storage)
        #print(f"TOTEX = {round(obj_opt,2)}")
        
        return(obj_opt)
    except Exception as e:
        return np.nan

# Define the range of values for T_OA and T_sup_HP
storage_range = np.linspace(6, 50, 15)
heatpump_range = np.linspace(2, 40, 15)

# Generate a grid of T_OA and T_sup_HP values
storage_grid, heatpump_grid = np.meshgrid(storage_range, heatpump_range)

# Calculate the Q_max for each combination of T_OA and T_sup_HP
load = [7]*24
cost_values = np.vectorize(get_TOTEX)(storage_grid, heatpump_grid)
load = [15]*24
cost_values2 = np.vectorize(get_TOTEX)(storage_grid, heatpump_grid)
#load = [20]*24
#cost_values3 = np.vectorize(get_TOTEX)(storage_grid, heatpump_grid)

min_index = np.unravel_index(np.nanargmin(cost_values), cost_values.shape)
min_storage_value = storage_grid[min_index]
min_heatpump_value = heatpump_grid[min_index]

# Create a 3D plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(projection='3d')

# Plot the surface
surface = ax.plot_surface(storage_grid, heatpump_grid, cost_values, cmap='viridis', alpha=0.7)
surface2 = ax.plot_surface(storage_grid, heatpump_grid, cost_values2, cmap='coolwarm', alpha=0.7)
#surface3 = ax.plot_surface(storage_grid, heatpump_grid, cost_values3, cmap='plasma', alpha=0.7)

# Add the minimum
#ax.scatter(min_storage_value, min_heatpump_value, cost_values[min_index], color='red', s=100)

# Add labels and title
ax.set_xlabel('Storage capacity [kWh]')
ax.set_ylabel('Heat pump capacity [kWh]')
ax.set_zlabel('Cost of optimal operation [$/day]')
ax.set_zticks([])
ax.set_title('Optimal operating cost for different equipement sizes')

# Show the plot
plt.show()
