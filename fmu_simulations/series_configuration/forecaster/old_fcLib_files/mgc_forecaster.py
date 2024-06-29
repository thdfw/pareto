import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import datetime as dtm

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    root = os.getcwd()

# Import custom packages
from fmlc import eFMU, controller_stack, check_error

fc_to_ctrl_map = {'fc-dhi': 'dhi',
                  'fc-ghi': 'ghi',
                  'fc-gni': 'gni',
                  'fc-oat': 'oat',
                  'grid_available': 'grid_available',
                  'fc-loadhi': 'load_shed_potential_hi',
                  'fc-loadlow': 'load_shed_potential_low',
                  'fc-pv': 'generation_pv'}

class forecaster_dummy(eFMU):
    def __init__(self):
        self.input = {'wf':None, 'scada':None, 'timeout': None}
        self.output = {'output-data':None}
        
        self.i = 0
    def compute(self):
        timeout = 3
        st = time.time()
        
        
        now = dtm.datetime.now()
        #print('START', now, self.output['data'])
        self.i += 1
        #if self.i > 2:
        #    time.sleep(6)
        #print('DONE', now, self.output['data'])
        
        wf = pd.read_json(self.input['wf'])
        wf['P_load'] = np.random.randint(5, 30, size=len(wf)) # Load, in kW
        wf['P_pv'] = np.sin(wf.index.hour/3-2.5) * 10
        wf['P_pv'] = wf['P_pv'].mask(wf['P_pv']<0, 0)
        # Scada
        res_cols = ['P_load','P_pv']
        
        
        scada = pd.read_json(self.input['scada']).set_index('name')
        
        for c in res_cols:
            wf.loc[wf.index[0], c] = scada.loc[c, 'value']
            
        # Check timeout before writing data
        
        # add: self.return_on_timeout()
        
        
#         if st+timeout < time.time():
#             for k in self.output.keys():
#                 self.output[k] = -1 # default
#             return f'Timed out after {round(time.time()-st, 1)} s.'
        
        
        
        
        self.output['output-data'] = wf[res_cols].to_json()
        return 'Done.'
    
class forecaster_scada(eFMU):
    def __init__(self):
        self.input = {'input-data': None, 'timeout': None, 'fc-map': None, 'tz': None, 'hours': None, 'freq': None}
        self.output = {'output-data': None, 'wf-data': None, 'time': None, 'duration': None}
        self.i = 0
    def compute(self):
        st = time.time()
        msg = ''
        now = self.input['time']
        forecast = pd.read_json(self.input['input-data'])
        #tz = int(forecast.loc['fc_timezone', 'value'])
        #forecast.loc['fc_timezone'] = np.nan
        tz = int(self.input['tz'])
        forecast = forecast.dropna()
        forecast.index = pd.MultiIndex.from_tuples([('-'.join(ix.split('_')[1].split('-')[:-1]),
                                             int(ix.split('-')[-1].split('_')[0])) for ix in forecast['name'].values])
        forecast = forecast['value'].unstack(0)
        #forecast.index = pd.MultiIndex.from_tuples([('-'.join(ix.split('_')[1].split('-')[:-1]),
        #                                             int(ix.split('-')[-1].split('_')[0]),
        #                                             int(ix.split('-')[-1].split('_')[1])) for ix in forecast['name'].values])
        #forecast = forecast['value'].unstack(1).unstack(1).transpose()
        forecast = forecast.sort_index()
        ix_start = pd.to_datetime(now, unit='s').tz_localize('UTC').tz_convert('Etc/GMT{0:+d}'.format(-1*tz)).tz_localize(None)
        ix_start = ix_start.replace(second=0, microsecond=0)


        hours = self.input['hours'] # 23
        freq = self.input['freq'] # 15
        ix = pd.date_range(ix_start.replace(minute=0), ix_start+pd.DateOffset(hours=hours, minutes=55), freq=f'{freq}T')
        ix = ix[ix >= ix_start]
        if ix[0] != ix_start:
            ix = ix.insert(0, ix_start)
        if len(ix) != len(forecast):
            forecast = forecast.iloc[:len(ix)]
        forecast.index = ix[:int((hours+1) * (60/freq))]

        
        
        
#         forecast.index = pd.MultiIndex.from_tuples([('-'.join(ix.split('_')[1].split('-')[:-1]),
#                                                      int(ix.split('_')[1].split('-')[-1])) for ix in forecast['name'].values])
#         forecast = forecast['value'].unstack().transpose()

#         ix_start = pd.to_datetime(now, unit='s').tz_localize('UTC').tz_convert('Etc/GMT{0:+d}'.format(-1*tz)).tz_localize(None)
#         ix_start = ix_start.replace(second=0, microsecond=0)
#         forecast.index = [ix_start+pd.DateOffset(hours=ix) for ix in forecast.index]
        
        forecast = forecast.rename(columns=self.input['fc-map'])
        self.output['output-data'] = forecast.rename(columns=fc_to_ctrl_map).to_json()
        wf = forecast[['fc-oat','fc-dhi','fc-gni','fc-ghi']].shift(0.5, freq='1H').resample('1H', \
            offset=pd.Timedelta(minutes=forecast.index[0].minute)).mean() # shift by half interval (30 min) to center align
        self.output['wf-data'] = wf.to_json() 
        self.output['time'] = str(forecast.index[0])
        self.output['duration'] = time.time() - st
        return 'Done.'
    
class merge_forecasts(eFMU):
    def __init__(self):
        self.input = {'fc-pv': None, 'fc-loadlow': None, 'fc-loadhi': None, 'fc-grid': None, 'timeout': None}
        self.output = {'output-data': None, 'duration': None}
        self.i = 0
        
    def check_and_merge(self, dfs):
        forecast = pd.DataFrame()
        none_in_dfs = False
        for fn in dfs:
            if self.input[fn] != None and self.input[fn] != -1:
                forecast = pd.concat([forecast, pd.read_json(self.input[fn]).rename(columns={'y': fn})], axis=1)
            else:
                none_in_dfs = True
        if not none_in_dfs:
            forecast.index = pd.to_datetime(forecast.index)
            return forecast
        else:
            return pd.DataFrame()
        
    def compute(self):
        st = time.time()
        self.msg = ''
        now = self.input['time']
        self.output['output-data'] = -1
        
        try:
            forecast = self.check_and_merge(['fc-pv', 'fc-loadlow', 'fc-loadhi'])
            forecast = forecast.rename(columns=fc_to_ctrl_map)
            if not forecast.empty:
                other = pd.read_json(self.input['fc-grid'])
                other.index = pd.to_datetime(other.index)
                other = other.shift(0.5, freq=forecast.index[1]-forecast.index[0]).resample(\
                    forecast.index[1]-forecast.index[0], offset=pd.Timedelta(minutes=forecast.index[0].minute)).mean()
                forecast = pd.concat([forecast, other[[c for c in other.columns if c not in forecast.columns]]], axis=1)                
                forecast = forecast.rename(columns=fc_to_ctrl_map)
                forecast = forecast.dropna()
                self.output['output-data'] = forecast.to_json()


    #         forecast = pd.read_json(self.input['input-data'])
    #         #tz = int(forecast.loc['fc_timezone', 'value'])
    #         #forecast.loc['fc_timezone'] = np.nan
    #         tz = int(self.input['tz'])
    #         forecast = forecast.dropna()
    #         forecast.index = pd.MultiIndex.from_tuples([('-'.join(ix.split('_')[1].split('-')[:-1]),
    #                                              int(ix.split('-')[-1].split('_')[0])) for ix in forecast['name'].values])
    #         forecast = forecast['value'].unstack(0)
    #         #forecast.index = pd.MultiIndex.from_tuples([('-'.join(ix.split('_')[1].split('-')[:-1]),
    #         #                                             int(ix.split('-')[-1].split('_')[0]),
    #         #                                             int(ix.split('-')[-1].split('_')[1])) for ix in forecast['name'].values])
    #         #forecast = forecast['value'].unstack(1).unstack(1).transpose()
    #         forecast = forecast.sort_index()
    #         ix_start = pd.to_datetime(now, unit='s').tz_localize('UTC').tz_convert('Etc/GMT{0:+d}'.format(-1*tz)).tz_localize(None)
    #         ix_start = ix_start.replace(second=0, microsecond=0)


    #         hours = self.input['hours'] # 23
    #         freq = self.input['freq'] # 15
    #         ix = pd.date_range(ix_start.replace(minute=0), ix_start+pd.DateOffset(hours=hours, minutes=55), freq=f'{freq}T')
    #         ix = ix[ix >= ix_start]
    #         if ix[0] != ix_start:
    #             ix = ix.insert(0, ix_start)
    #         if len(ix) != len(forecast):
    #             forecast = forecast.iloc[:len(ix)]
    #         forecast.index = ix[:int((hours+1) * (60/freq))]




    # #         forecast.index = pd.MultiIndex.from_tuples([('-'.join(ix.split('_')[1].split('-')[:-1]),
    # #                                                      int(ix.split('_')[1].split('-')[-1])) for ix in forecast['name'].values])
    # #         forecast = forecast['value'].unstack().transpose()

    # #         ix_start = pd.to_datetime(now, unit='s').tz_localize('UTC').tz_convert('Etc/GMT{0:+d}'.format(-1*tz)).tz_localize(None)
    # #         ix_start = ix_start.replace(second=0, microsecond=0)
    # #         forecast.index = [ix_start+pd.DateOffset(hours=ix) for ix in forecast.index]

    #         forecast = forecast.rename(columns=self.input['fc-map'])

        except Exception as e:
            self.msg += f'ERROR: Failed to generate forecasts: {e}\n\n{traceback.format_exc()}\n'
            
        if self.msg == '':
             self.msg = 'Done.'
                
        self.output['duration'] = time.time() - st
        return self.msg
    
scadaColumnList = ['weather-dhi', 'weather-gni', 'weather-ghi', 'weather-rhu',
                   'weather-oat', 'weather-wdi', 'weather-wsp']

# 'irn0_baseload-hi', 'irn0_baseload-low', 'irn0_power-pv'

def default_parameter_forecaster(targetName='irn0_power-pv', backupPath=None, inputPath=None,
                                 min_days=14, seed=1, horizon=24, stepsize=60*60, backupInterval=24*60*60,
                                 trainingInterval=24*60*60, train_method='train_test_split',
                                 trainingIntervalBest=24*60*60, min_score=0,
                                 scadaColumnList=scadaColumnList):
    cfg = {'inputPath': inputPath, # Initialize with database/csv. './forecaster_example_data_simple.csv',
           'columnList': [], # Only to init record; not used otherwise.
           'scadaColumnList': scadaColumnList, # Scada columns to store
           'targetName': targetName, # Column to extract from dataset as y.
           'unpackHourly': False, # Unpack hourly data into separate samples or keep as one sample
           'backupData': True, # Flag to automatically backup data.
           'backupPath': backupPath, # Backup filepath.
           'backupInterval': backupInterval, # Backup interval, in seconds.
           'trainingInterval': trainingInterval, # Training interval, in seconds.
           'trainingIntervalBest': trainingIntervalBest, # Training interval best only, in seconds.
           'predict_last': True,
           'min_score': min_score,
           'inputProcessed': False,
           'fcParams': {'train_size': 0.75, # split between train and test samples.
                        'train_method': train_method, # method to split samples (daily or train_test)_split.
                        'min_days': min_days, # min days before operation
                        #'colList': [], # not used
                        #'targetColName': None, # not used
                        #'recordPath': './data.csv', # not used
                        #'backupInterval': 7*24*60*60, # not used
                        #'unpackHourly': True, # not used
                        'horizon': horizon, # horizon of forecast data (all weather data is used; can be less than that)
                        'stepsize': stepsize, # stepsize of forecast in seconds
                        'add_features': True,
                        'seed': seed}} # seed
    return cfg