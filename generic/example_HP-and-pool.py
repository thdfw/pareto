from functions import generic
from poolstuff import get_modes

# ---------------------------------------------------
# Example 1: User inputs for a HPTES
# ---------------------------------------------------

parameters_HPTES = { 

    # Prediction horizon, in hours
    'horizon': 24,

    # Electricity prices for every hour in the horizon, in cts/kWh
    'elec_costs': [7.92, 6.63, 6.31, 6.79, 8.01, 11.58, 19.38, 21.59, 11.08, 4.49, 1.52, 0.74, 
                   0.42, 0.71, 0.97, 2.45, 3.79, 9.56, 20.51, 28.26, 23.49, 18.42, 13.23, 10.17],
    
    # The load can either be hourly or daily
    # When hourly, the values are given for every hour in the horizon, in kWh
    # When daily, a single value is given in kWh
    'load': {'type': 'hourly', 
             'value': [5.91, 5.77, 5.67, 5.77, 5.71, 6.06, 6.34, 6.34, 6.01, 5.77, 5.05, 5.05, 
                       4.91, 4.91, 4.91, 4.91, 5.05, 5.1, 4.91, 4.91, 4.91, 4.91, 4.98, 4.91]},

    # With a HP, the control is the hourly heating power (kW or kWh, both are equivalent with hourly time steps)
    # The control can be within a continuous operating range (i.e. it must be between a given minimum and a maximum value)
    # Or the control can be a choice between discrete operating modes
    'control': {'type': 'range',
                'max': [12]*24, #kW
                'min': [6]*24}, #kW
    
    # The constraints are activated using True instead of false, and have corresponding parameters
    'constraints': {'storage_capacity': True,   # If there is a storage limit
                    'max_storage': 30,          # The storage capacity, in kWh
                    'initial_soc': 0,           # Initial state of charge
                    'cheaper_hours': True,      # Leave True when working with hourly loads
                    'quiet_hours': False        # Assign hours during which the system must not operate
                    },

    'hardware': {'heatpump': True,              # True if there is a heat pump
                 'COP': [3]*24}                 # Estimated COP at every hour of the horizon, typically COP(T_OA)
}

# ---------------------------------------------------
# Example 2: User inputs for a pool pump
# ---------------------------------------------------

# Get required flow volume and hours ranked by price
required_flow_vol, hours_ranked = get_modes(parameters_HPTES['elec_costs'])

parameters_pool = { 
    'horizon': 24,

    'elec_costs': [7.92, 6.63, 6.31, 6.79, 8.01, 11.58, 19.38, 21.59, 11.08, 4.49, 1.52, 0.74, 
                   0.42, 0.71, 0.97, 2.45, 3.79, 9.56, 20.51, 28.26, 23.49, 18.42, 13.23, 10.17],
    
    'load': {
        'type': 'daily', 
        'value': required_flow_vol,
        },

    'control': {
        'type': 'mode',
        'hours_ranked': hours_ranked
        },
    
    'constraints': {
        'storage_capacity': False,                    
        'cheaper_hours': False,
        'quiet_hours': True,
        'quiet_hours_list': [0, 1, 2, 3, 4, 5, 6, 22, 23]
        },
    
    'hardware': {'heatpump': False}
}

# ---------------------------------------------------
# Get the sequence
# ---------------------------------------------------

control = generic(parameters_HPTES)
print(f"\nThe control sequence is:\n{control}\n")