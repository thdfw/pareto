import sys
import os
import logging
import pandas as pd
import numpy as np
import json
import traceback
import time
import datetime as dtm

from fmlc import eFMU

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
import fcSelector
import fcLib
import fcDataMgmt

# Setup logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s (%(name)s:%(funcName)s)',
    datefmt='%Y-%m-%d %H:%M:%S')

class ForecasterWrapper(eFMU):
    def __init__(self):
        self.input = {
            'forecaster-list': None, 
            'training-data': None,
            'weather-data': None,
            'scada-data': None,
            'data-timestamp': None,
            'input-data': None,
            'debug': None,
            'timeout': None
        }  
        
        self.output = {
            'model-summary': None,
            'output-data': None,
            'duration': None
        }
        
        # set init flag for data initialization
        self.init = False

        # initialize data manager
        self.dataManager = None
        self.csvMetaData = {}
        
        # initialize selector and forecaster attributes
        self.trainingDate = None
        self.trainingDateBest = None
        self.framework = None
        self.bestModel = None
        
        # initialize selected forecaster prediction
        self.fcPrediction = None
        
        self.debug = None 
        
        self.logger = logging.getLogger(__name__)
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)
        self.msg = ''
        
        
    def check_training(self):
        # determine if its the training hour
        trainingHour = self.now.hour == self.trainingHour
    
        # check if time for training all
        if self.trainingDate != None:
            trainingDelta = (self.now - self.trainingDate).total_seconds()
            retrainModels = (trainingDelta >= self.trainingInterval) and trainingHour
        else:
            retrainModels = False
            
        # check if time for training best
        if self.trainingDateBest != None:
            trainingDelta = (self.now - self.trainingDateBest).total_seconds()
            retrainBestOnly = (trainingDelta >= self.trainingIntervalBest) and trainingHour
        else:
            retrainBestOnly = False
            
        return retrainModels, retrainBestOnly
    
    
    def extractInputs(self):
        '''
        method extract all possible data fields from current value for self.inputs
        and stores them in corresponding class attributes
        '''
        
        self.inputData = self.input.get('input-data', {})
        self.now = pd.to_datetime(self.input.get('data-timestamp', dtm.datetime.now())).to_pydatetime()
        self.debug = self.input.get('debug', False)
        
        # extract training data inputs from self.input
        trainData = self.input.get('training-data', {})
        self._trainData = trainData

        # initialize data flags
        self.dataValid = None
        self.dataMethod = None

        # extract attributes from inputs
        self.dataX = trainData.get('dataX', None)
        self.dataY = trainData.get('dataY', None)

        self.newRecord = trainData.get('newRecord', {})
        
        self.inputPath = trainData.get('inputPath', None)
        self.csvMetaData = trainData.get('csvMetaData', self.csvMetaData)
        self.columnList = trainData.get('columnList', [])
        self.scadaColumnList = trainData.get('scadaColumnList', None)
        self.targetName = trainData.get('targetName', None)
        self.unpackHourly = trainData.get('unpackHourly', False)

        self.backupData = trainData.get('backupData', None)        
        self.backupPath = trainData.get('backupPath', None)
        self.backupInterval = trainData.get('backupInterval', None)
        
        self.trainingInterval = trainData.get('trainingInterval', 60*60*24*7 ) # default value 1 week
        self.trainingIntervalBest = trainData.get('trainingIntervalBest', 60*60*24*2 ) # default value 2 days
        self.trainingHour = trainData.get('trainingHour', 0) # default at midnight
        self.predict_last = trainData.get('predict_last', True) # predict based on last inputs
        self.minScore = trainData.get('min_score', 0) # min score for valid forecaster
        
        # check for flag that training data has already been processed on previous iteration
        self.inputProcessed = trainData.get('inputProcessed', False)
        self.fcParams = trainData.get('fcParams', {})
        
        # create data manager to read and process data
        self.mgmtParams = {
            'colList': self.columnList,
            'scadaColumnList': self.scadaColumnList,
            'targetColName': self.targetName,
            'backupData': self.backupData,
            'backupPath': self.backupPath,
            'backupInterval': self.backupInterval,
            'unpackHourly': self.unpackHourly,
            'now': self.now,
            'csvMetaData': self.csvMetaData
        }
        
        # extract training data inputs from self.input
        self.newRecord = trainData.get('newRecord', {})
        
        # check for new record fields in input
        self.dataScada = self.input.get('scada-data', None)
        self.dataWeather = self.input.get('weather-data', None)
        self.dataTs = self.input.get('data-timestamp', None)
        
        # check for update data in inputs
        self._scadaUpdate = (self.dataScada is not None and self.dataWeather is not None)
        self._simpleUpdate = self.newRecord is not None
        self._doUpdate = self._scadaUpdate or self._simpleUpdate
        
        return
       
    def compute(self):
        
        # initialize msg to return
        st = time.time()
        self.msg = ''
        self.newPrediction = False
        self.valid = True
        
        self.fcList = json.loads(self.input.get('forecaster-list', '[]'))
       
        # extract all current inputs from self.inputs
        self.extractInputs()       
        
        try:
            # run data initialization. only executes on first compute call
            self.initData()
            # update data based on current value of self.input
            
            # update data manager
            self.dataManager.updateParams(self.mgmtParams)
            
            # check for input data in training data
            if self._doUpdate:
                self.updateInputs()
                self.msg += self.dataManager.msg
        except Exception as e:
            self.msg += f'ERROR: Failed to hande input data.\nError: {e}\n\n{traceback.format_exc()}\n'
            self.valid = False
        
        if self.valid:
            if (self.trainingData['X'] is None) or (self.trainingData['y'] is None):
                self.msg += f'ERROR: Failed to generate training data.\n'
            else:
                # calculate time since last training (if previous training exists)
                retrainModels, retrainBestOnly = self.check_training()

                # if framework selection has not been made, run evaluate method
                if self.bestModel == None or retrainModels or retrainBestOnly:
                
                    # if model hasn't been selected yet, run evaluate method
                    try:
                        bestonly = True
                        # reduce forecasters if bestonly
                        fcList = self.fcList
                        if self.bestModel and retrainBestOnly and not retrainModels:
                            fcList = [fc for fc in self.fcList if fc['name'] == self.bestModel['name']]
                        # initialize forecastor selection framwork
                        self.framework = fcSelector.ForecasterFramework(params=self.fcParams,
                                                                        data=self.trainingData, 
                                                                        fcList=fcList,
                                                                        debug=self.debug)
                        self.framework.evaluateAll()
                        self.msg += self.framework.msg
                        self.valid = self.valid and self.framework.validData
                        
                        if self.valid and self.framework.best['score_rmse'] > self.minScore:
                            self.bestModel = self.framework.best.to_dict()
                            self.trainingDateBest = self.now
                            if retrainModels or not self.trainingDate:
                                self.trainingDate = self.now
                     
                    except Exception as e:
                        self.msg += f'ERROR: Failed to train forecasters in fc list.\nForecaster message: {self.msg}\nError: {e}\n\n{traceback.format_exc()}\n'
                    
                # run prediction on fcData if passed
                try:
                    # run prediction method of best
                    if (self.inputData is not None) and self.bestModel:
                        self.fcPrediction = pd.Series(self.bestModel['model'].predict(self.inputData).tolist())
                        self.newPrediction = True
                    elif self.predict_last and self.bestModel:
                        X = self.trainingData['X'].iloc[-1:].copy()
                        X.loc[X.index, 'obs_y'] = self.trainingData['y'].loc[X.index] # add current y to obs
                        X = self.framework.add_index_features(X) # add index features to obs
                        self.fcPrediction = pd.DataFrame(self.bestModel['model'].predict(X)).transpose()
                        self.fcPrediction.index = [X.index[0] + pd.DateOffset(seconds=i*self.framework.stepsize) for i in range(len(self.fcPrediction))]
                        self.fcPrediction.columns = ['y']
                        self.newPrediction = True
                    elif self.predict_last and not self.bestModel:
                        self.msg += 'WARNING: Waiting to initialize forecaster.\n'
                except Exception as e:
                    self.msg += f'ERROR: Failed to generate new prediction: {e}\n\n{traceback.format_exc()}\n'
                    self.newPrediction = False
            
        # package results into self.output
        if self.bestModel:
            self.output['model-summary'] = {'lastTrainedBest': str(self.trainingDateBest),
                                            'lastTrainedAll': str(self.trainingDate),
                                            'bestModelName': self.bestModel['name'],
                                            'bestScore': self.bestModel['score'],
                                            'bestScoreAdj': self.bestModel['score_adj'],
                                            'bestScoreMse': self.bestModel['score_mse'],
                                            'bestScoreRmse': self.bestModel['score_rmse']}
        else:
            self.output['model-summary'] = {'lastTrainedBest': str(self.trainingDateBest),
                                            'lastTrainedAll': str(self.trainingDate)}
        
        if self.newPrediction:
            self.output['output-data'] = self.fcPrediction.to_json() #json.dumps(self.fcPrediction.__dict__)
        else:
            self.output['output-data'] = None

        # if no msg has been define, default to Done
        if self.msg == '':
             self.msg = 'Done.'
             
        
        self.output['duration'] = time.time()-st
        
        # Return status message
        return self.msg


    def initData(self):
        '''
        method initializes training data based on items passed in 'training-data'.
        only executes once, based on value of self.init

        does the following operations
        1 - if dataX and dataY in inputs, uses them directly
        2 - if inputPath and targetName in inputs, reads X,y data from file
        3 - if only target provided, initializes an empty X,y

        '''
        
        # if self.init flag True, return
        if self.init:
            return
        
        # if no training data provided in input, use existing dataset
        if self._trainData == {}:
            return
            
        # method 1: inputs contain 'trainingg-data'; use X and y directly
        if (self.dataX is not None) and (self.dataY is not None):
            
            print('using X and y directly')
            
            self.dataMethod = 'direct'

            self.trainingData = {
                'X': self.dataX,
                'y': self.dataY
            }
        
        else:

            # initialize data manager
            self.dataManager = fcDataMgmt.DataManager(params=self.mgmtParams, debug=self.debug)

            # method 2: inputs contain 'input-path' and 'target-column': read from file
            if self.inputPath is not None:
                # read and unpack data from csv
                self.dataManager.readRecord(self.inputPath)
                self.dataMethod = 'fromCsv'
            # method 3: if no inputs provided, proceed with empty data manager
            else:
                self.dataMethod = 'initEmpty'
                
            

            # add error msg if generated
            self.msg += self.dataManager.msg

            # package processed data for use by Selector
            self.trainingData = {
                'X': self.dataManager.X,
                'y': self.dataManager.y
            }
           
        
        # update init flag to true (wrapper has now been initialized)
        self.init = True
        
        return
    

    def updateInputs(self):
        '''
        Method adds new observation (row) to existing training dataset
        
        New record can be added either through top-level inputs:
            
        'scada-data' - str of json of scada reading
        'weather-data' - dataframe of weather forecast
        'data-timestamp' - optional - timestamp to use for observation
        
        If timestamp not provided, the first timestamp in the weather-data
        dataframe will be extracted and used for the official timestamp of 
        the new observation
        
        A simplified new record can also be passed as a dict inside the
        'training-data' input.
        
        '''
        
        # check that dataManager already exists
        if self.dataManager is None:
            self.msg += 'ERROR: Cannot add record. Data manager not configured'
            return
        
        # update time
        self.dataManager.now = self.now

        # option 1: if SCADA/Weather/Timestamp provided, combine and process
        # if dataScada is not None and dataWeather is not None and dataScada != -1 and dataWeather != -1:
        if self._scadaUpdate:
            
            # extract timestamp if dataTs has not been provided in inputs
            #if dataTs is None:
            #    # use index of first row of weather data
            #    try:
            #        dataTs = dataWeather.index[0]
            #    except Exception as e:
            #        self.msg += f'ERROR: Failed to extract timestamp from record: {e}\n\n{traceback.format_exc()}\n'
        
            # run method to combine input fields, add to dataset, and process
            self.dataManager.combineDataRecords(self.dataScada, self.dataWeather, self.dataTs, writeRecord=True)
            
            # package new X, y
            self.trainingData = {
                'X': self.dataManager.X,
                'y': self.dataManager.y
            }
            
            self.dataMethod = 'addedRecordCombined'
            
        
        # option 2: single record passed in newRecord inputs
        elif self._simpleUpdate:
            
            # run method to combine input fields, add to dataset, and process
            self.dataManager.addRecord(self.newRecord)   
            
            # package new X, y
            self.trainingData = {
                'X': self.dataManager.X,
                'y': self.dataManager.y
            }
     
            self.dataMethod = 'addedRecordSimple'
            
        else:
            self.valid = False
            self.msg += 'INFO: Waiting to initialize inputs.'
            
        # run back-up check if a dataManager exists
        if self.dataManager is not None:
            self.logger.info('running backup check in wrapper')
            self.dataManager.checkRecordBackup()
            
        
        return



if __name__ == "__main__":
    
    # load training data   
    data = pd.read_csv('forecaster_example_data.csv', index_col = 0)
    data.index = pd.to_datetime(data.index)
    
    # Split the data into X and y
    X_columns = [col for col in data.columns if not 'Ppv_forecast' in col]
    y_columns = 'Ppv_forecast_1'
    # y_columns = [col for col in data.columns if 'Ppv_forecast' in col]
    
    X = data[X_columns]
    y = data[y_columns]
    
    # extract 2 series from X and y for predictions
    Xtrain = X.iloc[:-48]
    Xpredict1 = X.iloc[-48:-24]
    Xpredict2 = X.iloc[-24:]
    
    ytrain = y.iloc[:-48]
    ypredict1 = y.iloc[-48:-24]
    ypredict2 = y.iloc[-24:]
    
    # package data for framework
    data_eval = {
        'X': Xtrain,
        'y': ytrain
    }
    
    # create a list of forecaster candidate to evalute
    fcList = fcLib.forecaster_list
    
    # selecting 4 arbitrary forecaster options
    fcListUser = [fcList[ii] for ii in [2,8,9,11]]

    # instantiate forecast framework wrapper
    n1 = ForecasterWrapper()
    
    # # update input data
    # newInputs = {
    #     'forecaster-list': fcListUser, 
    #     'training-data': {
    #         # direct data input
    #         'dataX': Xtrain, 
    #         'dataY': ytrain,
    #     },
        
    #     'input-data': Xpredict1
    # }
    
    
    ## case 2 - read directly from csv file
    
    # load training data   
    data = pd.read_csv('forecaster_example_data_simple.csv', index_col = 0)
    data.index = pd.to_datetime(data.index)
    
    # Split the data into X and y
    X_columns = [col for col in data.columns if not 'Ppv_forecast_1' in col]
    y_columns = 'Ppv_forecast_1'
    # y_columns = [col for col in data.columns if 'Ppv_forecast' in col]
    
    X = data[X_columns]
    y = data[y_columns]
    
    # extract 2 series from X and y for predictions
    Xtrain = X.iloc[:-48]
    Xpredict1 = X.iloc[-48:-24]
    Xpredict2 = X.iloc[-24:]
    
    ytrain = y.iloc[:-48]
    ypredict1 = y.iloc[-48:-24]
    ypredict2 = y.iloc[-24:]
    
    
    # # construct meta data descriptions
    # dataMeta = {
    #     'Tamb_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'clear_sky_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'cloud_cover_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'Ppv_dminus1_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'Ppv_forecast': {'tsIndexed': True, 'tsCount':23},
    # }
    
    # update input data
    newInputs = {
        'forecaster-list': fcListUser, 
        'training-data': {
            # direct data input
            'inputPath': './forecaster_example_data_simple.csv',
            'columnList': list(data.columns),
            'targetName': 'Ppv_forecast_1',
            'unpackHourly': False
        },
        
        'input-data': Xpredict1
    }
    
    n1.input = newInputs
    
    # run compute method to train and predict
    # n1.compute()
    
    
    ## case 3 - init empty dataset. add observations
    
    # load training data   
    data = pd.read_csv('forecaster_example_data_simple.csv', index_col = 0)
    data.index = pd.to_datetime(data.index)
    
    # Split the data into X and y
    X_columns = [col for col in data.columns if not 'Ppv_forecast_1' in col]
    y_columns = 'Ppv_forecast_1'
    # y_columns = [col for col in data.columns if 'Ppv_forecast' in col]
    
    X = data[X_columns]
    y = data[y_columns]
    
    # extract 2 series from X and y for predictions
    Xtrain = X.iloc[:-48]
    Xpredict1 = X.iloc[-48:-24]
    Xpredict2 = X.iloc[-24:]
    
    ytrain = y.iloc[:-48]
    ypredict1 = y.iloc[-48:-24]
    ypredict2 = y.iloc[-24:]
    
    
    # # construct meta data descriptions
    # dataMeta = {
    #     'Tamb_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'clear_sky_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'cloud_cover_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'Ppv_dminus1_forecast': {'tsIndexed': True, 'tsCount':23},
    #     'Ppv_forecast': {'tsIndexed': True, 'tsCount':23},
    # }
    
    # update input data
    newInputs = {
        'forecaster-list': fcListUser, 
        'training-data': {
            # direct data input
            # 'inputPath': './forecaster_example_data_simple.csv',
            # 'columnList': list(data.columns),
            'targetName': 'Ppv_forecast_1',
            'unpackHourly': False
        },
        
        'input-data': Xpredict1
    }
    
    n1.input = newInputs
    
    # run compute method to train and predict
    # n1.compute()
    
    
    newRecords = [
        {
            '2018-07-21 00:00:00': {
                'c1': 11,
                'c2': 22,
                'c3': 1,
                't': 101,
                'ts1': [1,2,3,4,5],
                'ts2': [24,25,26,27,28]
            }
        },
        {
            '2018-07-21 01:00:00': {
                'c1': 13,
                'c2': 25,
                'c3': 2,
                't': 103,
                'ts1': [11,2,3,4,5],
                'ts2': [24,25,26,27,28]
            }
        },
        {
            '2018-07-21 02:00:00': {
                'c1': 15,
                'c2': 35,
                'c3': 4,
                't': 112,
                'ts1': [12,2,3,4,5],
                'ts2': [24,25,26,27,28]
            }
        },
        {
            '2018-07-21 03:00:00': {
                'c1': 16,
                'c2': 44,
                'c3': 5,
                't': 124,
                'ts1': [13,2,3,4,5],
                'ts2': [24,25,26,27,28]
            }
        },
        {
            '2018-07-21 04:00:00': {
                'c1': 17,
                'c2': 49,
                'c3': 6,
                't': 144,
                'ts1': [14,2,3,4,5],
                'ts2': [24,25,26,27,28]
            }
        },
        {
            '2018-07-21 05:00:00': {
                'c1': 17,
                'c2': 49,
                'c3': 6,
                't': 144,
                'ts1': [14,2,3,4,5],
                'ts2': [24,25,26,27,28]
            }
        },
        {
            '2018-07-21 06:00:00': {
                'c1': 17,
                'c2': 49,
                'c3': 6,
                't': 144,
                'ts1': [14,2,45,56,77],
                'ts2': [24,25,26,27,28]
            }
        }
    ]
    
    
    # function for generating example scada observation
    def example_scada():
        data = {'soc': 0.5, 'pmax': 1e3, 'emax': 5e3, 
                'echa': 0.96, 'edis': 0.96, 'smin': 0.1, 'smax': 1.0}
        scada1 = pd.DataFrame(index=range(len(data)*3), columns=['name', 'value', 'valid', 'error'])
        ix = 0
        for par, v in data.items():
            for i in range(3):
                scada1.loc[ix, :] = [f'irn{i}_battery-{par}', v, 1, '']
                ix += 1
        return scada1.to_json()
    
    newScadaRecord = example_scada()
    
    # read example weather forecast from external file
    weatherFilePath = os.path.abspath(os.path.join(os.getcwd(), 
                    'example_weatherforecast.csv'))

    newWeatherRecord = pd.read_csv(weatherFilePath, index_col=0)
    
        
        
    # update input data
    newInputs = {
        'forecaster-list': fcListUser, 
        'training-data': {
            'targetName': 'irn0_battery-edis',
            'backupData': True,
            'backupPath': './TEST_backup.csv',
            'backupInterval': 5
        }        
    }
    
    n1.input = newInputs
    
    # run compute method to train and predict
    n1.compute()
    
    for rr in range(0,10):
        
        time.sleep(6)
        
        # create generic hypothetical timestamp
        dataTs=f'2018-07-21 {rr:02}:00:00'
    
        newInput2 = {
            'forecaster-list': fcListUser, 
            'data-timestamp':  dataTs,
            'scada-data': newScadaRecord,
            'weather-data': newWeatherRecord,
            'training-data': {
                'targetName': 'irn0_battery-edis',
                'backupData': True,
                'backupPath': './TEST_backup.csv',
                'backupInterval': 5
            }
        }
        
        n1.input = newInput2
        
        # run compute method to train and predict
        n1.compute()

    
