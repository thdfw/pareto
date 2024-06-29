import os
import sys
import copy
import time
import json
import logging
import pandas as pd
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import traceback

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#logging.basicConfig(
#    format='%(asctime)s %(levelname)-8s %(message)s',
#    level=logging.INFO,
#    datefmt='%Y-%m-%d %H:%M:%S')

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    root = os.getcwd()

# Append src directory to import forecaster library files
sys.path.append(os.path.join(root, '..', 'src'))
import fcLib

def adjust_r2(r2, n, p):
    return 1-(1-r2)*(n-1)/(n-p-1)
    
# default config
DEFAULT_CONFIG = {
    'train_size': 0.75,
    'train_method': 'train_test_split',
    'min_days': 5,
#    'colList': [],
#    'targetColName': None,
#    'recordPath': './data.csv',
#    'backupInterval': 7*24*60*60,
#    'unpackHourly': True,
    'horizon': 24, # number of steps
    'stepsize': 60*60, # step size in seconds
    'seed': int(time.time()),
    'max_workers': None,
    'add_features': True,
    'now': None,
    }

class ForecasterFramework():
    '''
    This class provides a framework for defining, training and
    evaluating a list of forecaster models (as defined in the)
    fcLib.py module.
    
    methods:
    __init__: Initializes the framework with a list of models to 
              evaulate, and a training dataset to use for model
              training and testing.
    updateData: allows users to pass new training data, which is
              split into X and y datasets
    updateForecasters: allows user to update list of forecast models
              and parameters for use in evaluation.
    evaluateAll: method to train/test all models using multi-processing
              or using simple loop, if parallel arg False
    selectFc: method to select best-performing forecast model based on
              results of evaluation
    scoreSummary: provides summary of each model performance
    plotPredictions: generates a plot of each model for testing horizon
    
              
    '''
    
    def __init__(self, params={}, data=None, fcList=[], debug=False, uid='root'):
        '''
        Initializes the framework with a list of models to 
        evaulate, and a training dataset to use for model
        training and testing.
        
        inputs:
        fcList: list of dicts defining forecaster models
            to evaluation. Dicts contain keys:
            
            fun: name of forecast model to evaluate, 
            corresponding to model types defined in fcLib
            
            name: (optional) string with name of forecast model
            being used. Necessary if passing multiple forecasters
            of same type with different parameters. Default name
            set to 'fun' name. If multple forecasters of same type
            passed, the 'name' will be auto incrememted
            
            parameters: parameters to tune each model.
        data: dict with X and y items, corresponding to features
            and target values. X should be pandas df, y numpy array

        '''
        
        self.params = copy.deepcopy(DEFAULT_CONFIG)
        self.params.update(params)
        self.fcList = fcList
        self.debug = debug
        self.uid = uid
        
        self.dataDf = None
        self.dataDfLong = None
        self.backupDate = datetime.now()
        self.dataMeta = {} # dict of meta-data describing training data fields
        
        self.fcData = None
        self.bestModel = None
        self.bestScore = None
        self.bestModelName = None
        self.trainingDate = None

        self.logger_name = f'{uid}:{__name__}'
        self.logger = logging.getLogger(self.logger_name)
        if self.debug:
            if self.debug == 'info':
                self.logger.setLevel(logging.INFO)
            elif self.debug == 'warning':
                self.logger.setLevel(logging.WARNING)
            else:
                self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)
        self.msg = ''

        self.dtFormat = '%Y-%m-%d %H:%M:%S' #'2018-07-21 01:00:00'
        
        
        '''
        self.dtFormat = None
        self.addNewCols = True
        self.showWarnings = False
        
        '''
        # parse parameters
        self.train_size = self.params['train_size']
        self.train_method = self.params['train_method']
        self.min_days = self.params['min_days']
#        self.colList = self.params['colList']
#        self.targetColName = self.params['targetColName']
#        self.recordPath = self.params['recordPath']
#        self.backupInterval = self.params['backupInterval']
#        self.unpackHourly = bool(self.params['unpackHourly'])
        self.horizon = self.params['horizon']
        self.stepsize = self.params['stepsize']
        self.seed = self.params['seed']
        if not self.params['max_workers']:
            self.max_workers = os.cpu_count()-2
        else:
            self.max_workers = int(self.params['max_workers'])
        self.addFeatures = bool(self.params['add_features'] )
        
        if self.params['now']:
            self.now = datetime.strptime(self.params['now'], self.dtFormat)
        else:
            self.now = datetime.now()
        
        # try:
        #     self.now = pd.to_datetime(self.params.get('now', dtm.datetime.now())).to_pydatetime()
        # except:
        #     self.now = pd.to_datetime(dtm.datetime.now()).to_pydatetime()
        

        # make sure backupInterval is int
#        try:
#            self.backupInterval = int(self.backupInterval)
#        except:
#            # if int conversion fails, default to 1 week
#            self.backupInterval = 7*24*60*60
        
        if data:
            self.updateData(data)
        else:
            self.initRecord()   


    # data & setting definition methods
    def validateData(self, add_days=0):
        '''method validates training data. may add additional tests'''
        try:
            data = self.data

            assert len(data['X']) == len(data['y']), "Length of data differs."
            
            if data['X'].index[0] + pd.DateOffset(days=self.min_days+add_days) > data['X'].index[-1]:
                self.logger.warning(f'Data horizon is too short.')
                self.msg += 'WARNING: Data horizon is too short.\n'
                self.validData = False
            else:
                self.validData = True 

        except Exception as e:
            self.logger.warning(f'Data validation failed')
            self.msg += f'WARNING: Data validation filed: {e}\n\n{traceback.format_exc()}\n'
            self.validData = False
            
    def add_index_features(self, X):
        X['dayofweek'] = X.index.dayofweek
        X['hourofday'] = X.index.hour
        X['daytype'] = X['dayofweek'].apply(lambda x: 0 if x < 5 else 1)
        return X
        
    def make_samples(self, data, horizon=24, stepsize=60*60, addFeatures=False):
        X = data['X'].copy(deep=True)
        
        horizon_tot = int(horizon * (60*60)/stepsize)
        
        for ix in X.index:
            # add current observation to X
            X.loc[ix, 'obs_y'] = data['y'].loc[ix]
            for h in range(horizon_tot):
                next_ix = ix + pd.DateOffset(seconds=stepsize*(h+1)) # +1 to start with next not current
                if next_ix in data['y'].index:
                    # X.loc[ix, f'y_{h}'] = data['y'].loc[next_ix]
                    try:
                        X.loc[ix, f'y_{h}'] = data['y'].loc[next_ix]
                    except:
                        
                        try:
                            X.loc[ix, f'y_{h}'] = data['y'].loc[next_ix].values[0]
                        except:
                            print(next_ix)
                            pass

                else:
                    X.loc[ix, f'y_{h}'] = np.nan
                    
        # add index features
        if addFeatures:
            X = self.add_index_features(X)
        
        # filter missing data
        X = X.dropna()
        
        # sort data
        X = X.sort_index()
        
        # split data
        y_cols = [f'y_{h}' for h in range(horizon_tot)]
        return {'X': X[[c for c in X.columns if c not in y_cols]],
                'y': X[y_cols]}
        
    def updateData(self, data):
        '''
        method allows user to replace training data 

        Inputs:
        data: dict with 'X' and 'y' items
        '''
        self.data = data
        self.X = None
        self.y = None
        
        self.validateData(add_days=1) # add one more day to allow for forecast horizon

        if self.validData:
            self.data = self.make_samples(data,
                                          horizon=self.horizon,
                                          stepsize=self.stepsize,
                                          addFeatures=self.addFeatures)
            self.validateData() # validate again with reshaped data
            
            if self.validData:
                self.X = self.data['X']
                self.y = self.data['y']
        
    def updateForecasters(self, fcList):
        '''
        method allows user to replace forecaster list

        Inputs:
        list of fcLib forecaster dict objects
        '''
        
        self.fcList = fcList
    
    # evulauation & prediction methods    
    def evaluate(self, model, X=None, y=None, random_state=None, train_size=0.75, train_method='train_test_split',
                 X_train=None, X_test=None, y_train=None, y_test=None):
        '''
        This method evaluates the model's ability to match a data set and returns
        the resulting r^2 value. It uses existing scikit-learn functions to do so.
        
        Inputs:
        X_train: Training subset of the input data.
        X_test: Testing subset of the input data.
        y_train: Training subset of the output data.
        y_test: Testing subset of the output data.
        
        Outputs:
        Returns the r^2 value evaluating the model's ability to fit the test
            data set.
        '''
        # init run status vars
        msg = ''
        default_output = False
        if X is None or y is None:
            # if full data is not provided, ensure train/test data is passed as input
            assert X_train is not None, "training data must be provided if full data is missing"
            assert y_train is not None, "training data must be provided if full data is missing"
            assert X_test is not None, "testing data must be provided if full data is missing"
            assert y_test is not None, "testing data must be provided if full data is missing"
            # use testing dataset for evaluation prediction
            #X_predict = X_test
        else:
            # else, use train_test_split to generate split datasets
            if len(y) > 1:
                if train_method == 'train_test_split':
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                                        random_state=random_state)
                if train_method == 'daily_split':
                    X_date = np.unique(X.index.date)
                    y_date = np.unique(y.index.date)
                    X_train, X_test, y_train, y_test = train_test_split(X_date, y_date, train_size=train_size,
                                                                        random_state=random_state)
                    X_train = X.loc[np.isin(X.index.date, X_train)]
                    X_test = X.loc[np.isin(X.index.date, X_test)]
                    y_train = y.loc[np.isin(y.index.date, y_train)]
                    y_test = y.loc[np.isin(y.index.date, y_test)]
                    
            # use full X dataset for evaluation prediction
            #X_predict = X
        
        # create fitted model, score, and prediction
        st = time.time()
        
        try:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test) 
            prediction = model.predict(X_test)
            mse = mean_squared_error(y_test, model.predict(X_test))
            p = 1 if type(y_train)==type(pd.Series(dtype=int)) else len(y_train.columns)
            n = len(np.unique(y_train.index.date))
            validData = True
            self.validData = True
        except Exception as e:
            #traceback.print_exc()
            msg += f'ERROR: {model}\n{e}\n\n{traceback.format_exc()}\n'
            default_output = True
            
        if default_output:
            validData = False
            score = -1
            prediction = []
            mse = -1
            p = 1
            n = 1
            
        res = {'model': model,
               'score': score if validData else -1,
               'score_adj': adjust_r2(score, n, p) if validData else -1,
               'score_mse': mse if validData else -1,
               'score_rmse': np.sqrt(mse) if validData else -1,
               'x-cols': json.dumps(list(X_train.columns)),
               'y-cols': json.dumps(list(y_train.columns)),
#               'y-cols': json.dumps(y.name),
               'prediction': prediction,
               'duration': time.time() - st,
               'msg': msg}
        return res
        
    def evaluateAll(self, parallel=False):
        '''
        method evaluates all forecaster models in self.fcList using training data stored
        in self.X and self.y

        Inputs:
        parallel (bool): options to use multi-thread (concurrent.futures) when training
        ''' 

        # if record data not valid, skip eval
        if self.X is None or self.y is None:

            msg = 'No training data or target column provided. Evaluation suspended.'
            self.logger.warning(msg)
            #self.msg = msg

            return
        
        # set traing time to current 'now'
        self.trainingDate = self.now
        
        # initialize dict of forecaster data
        fcData = []
        predictions = []
        # initialize list of forecaster futures (for multiprocessing)
        modelFutures = {}
        
        # filter scada != valid data
        valid_ix = self.X[self.X['dataValid']].index
        X = self.X.loc[valid_ix].copy()
        y = self.y.loc[valid_ix].copy()
    
        # Split data before evaluation
        if len(y) > 1:
        
            if self.train_method == 'train_test_split':
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size,
                                                                    random_state=self.seed)
            elif self.train_method == 'daily_split':
                X_date = np.unique(X.index.date)
                y_date = np.unique(y.index.date)
                X_train, X_test, y_train, y_test = train_test_split(X_date, y_date, train_size=self.train_size,
                                                                    random_state=self.seed)
                X_train = X.loc[np.isin(X.index.date, X_train)]
                X_test = X.loc[np.isin(X.index.date, X_test)]
                y_train = y.loc[np.isin(y.index.date, y_train)]
                y_test = y.loc[np.isin(y.index.date, y_test)]
                
            else:
                raise ValueError(f'Training method "{self.train_method}" not available.')
                
        if len(X_train) >= self.min_days:
            self.validData = False
            
            # store X, y
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            # loop through forecasters given by user
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(os.cpu_count()-2, self.max_workers)) as executor:
                for i, fc in enumerate(self.fcList):
                    try:
                        # import and instantiate forecaster
                        fcObj = getattr(fcLib, fc['fun'])(**fc['parameter'])

                        # create run evaluate method of each forecaster as multiprocess
                        self.logger.info(f'evaluating model {fc["name"]}')

                        if parallel:
                            #modelFutures[fc['name']] = executor.submit(self.evaluate, fcObj, self.X, self.y,
                            #                                          self.seed, self.train_size, self.train_method)
                            modelFutures[fc['fun']] = executor.submit(self.evaluate, fcObj, None, None,
                                                                      self.seed, self.train_size, self.train_method,
                                                                      X_train, X_test, y_train, y_test)
                        else:
                            st = time.time()
                            #results = self.evaluate(fcObj, X=self.X, y=self.y, random_state=self.seed,
                            #                        train_size=self.train_size, train_method=self.train_method)
                            results = self.evaluate(fcObj, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                                    random_state=self.seed)
                            results['name'] = fc['name']
                            predictions.append(results['prediction'])
                            del results['prediction']
                            fcData.append(results)
                            
                    except Exception as e:
                        #traceback.print_exc()
                        msg = f"Failed to instantiate forecaster {fc['name']} with {e}\n\n{traceback.format_exc()}\n"
                        self.logger.error(msg)
                        self.msg += msg
                        
            if parallel:
                for i, fc in enumerate(self.fcList):
                    results = modelFutures[fc['fun']].result()
                    results['name'] = fc['name']
                    predictions.append(results['prediction'])
                    del results['prediction']
                    fcData.append(results)

            self.fcData = pd.DataFrame(fcData)
            self.predictions = predictions
            
            self.selectFc()
            
            for msg in self.fcData['msg'].values:
                if msg != '':
                    self.msg += '\n'+msg

        else:
            self.msg += 'WARNING: Note enough samples'

        # run data backup method
        # self.checkRecordBackup()
    
    def selectFc(self):
        '''
        method selects best model from trained forecast candidate models
        '''

        self.best = self.fcData.sort_values('score').iloc[-1]
        self.bestModel = self.best['model']
        self.bestScore = self.best['score']
        self.bestModelName = self.best['name']
    
    def predict(self, X):
        '''
        methods generates prediction using selected best model

        Inputs:
        X - dataframe with same columns/features as training data, indexed over forecast horizon
        '''
        assert self.bestModel is not None, 'Forecaster models must be fit before prediction.'
        return self.bestModel.predict(X)            
    
    def plotPredictions(self):
        '''
        method generates plot of predicts for all trained candidate models
        '''
        
        assert self.fcData is not None, 'Forecaster models must be fit before plotting'
        
        plt.figure(figsize = (10, 6))
        plt.plot(self.y.index, self.y, label = 'target', linestyle = 'dashed')
        
        for ix, fcResults in self.fcData.iterrows():
            fcLabel = f'{fcResults["name"]} - {fcResults["score"]:0.3f} - {fcResults["score_adj"]:0.3f}'
            plt.plot(self.y.index, self.predictions[ix], label = fcLabel)
        plt.legend()
        
        plt.ylabel(self.y.name)
        plt.tight_layout()
        plt.show() 

    # # data management methods
    # def unpackRecord(self):
    #     '''
    #     method unpacks record df into X and y using self.targetColName

    #     if unpackHourly is enabled, method will reshape training data to create
    #     more observations for forecasted input (e.g. hourly temperature forecasts)
    #     '''
        
    #     # init data attr to empty
    #     self.data = {}
    #     self.X = None
    #     self.y = None
        
    #     # if no data stored, do nothing
    #     if self.dataDf is None:
    #         return
        
    #     if self.dataDf.shape[0] == 0:
    #         return

    #     # check if unpackHourly is enabled
    #     if self.unpackHourly:

    #         # if no column metadata provided, do nothing
    #         if self.dataMeta == {}:
    #             return

    #         # use reshapeTsData method to unpack hourly columns
    #         self.reshapeTsData()
    #         dataDf = self.dataDfLong.copy()

    #     # otherwise, use self.dataDf.columns directly
    #     else:
    #         dataDf = self.dataDf.copy()

    #     #  drop rows with NaN values
    #     dataDf.dropna(inplace=True) 


    #     # get list of all columns
    #     dfColsList = list(dataDf.columns)

    #     # check if targetCol is currently in df
    #     if self.targetColName is None or self.targetColName not in dfColsList:
    #         # if target is not in list of column, use last column from df
    #         msg = f"Target column {self.targetColName} not found in data"
    #         logging.warning(msg)

    #         self.msg = msg

    #         # if no target provided, set data to None so that forecaster cannot be run
    #         self.data = {
    #             'X': None,
    #             'y': None
    #         }

    #         self.X = None
    #         self.y = None

    #         return 

    #     # if target column is defined, proceed with data unpacking
    #     targetColName = self.targetColName

    #     # divide data based on current targetColName
    #     dfColsList.remove(targetColName)
    #     columnsX = dfColsList
    #     columnsY = targetColName

    #     self.data = {
    #         'X': dataDf[columnsX],
    #         'y': dataDf[columnsY]
    #     }

    #     self.X = dataDf[columnsX]
    #     self.y = dataDf[columnsY]

    #     self.validateData()

    # def initRecord(self):
    #     '''
    #     method initializes record dataframe. Creates empty dataframe
    #     '''
        
    #     colList = self.colList

    #     # add columns to self.dataMeta
    #     if colList is not None:
    #         for col in colList:
    #             self.dataMeta[col] = {}

    #     # remove timestamp from list, which is used as index
    #     if 'timestamp' in colList:
    #         colList.remove('timestamp')

    #     # initialize df
    #     df = pd.DataFrame(columns = colList)

    #     self.dataDf = df

    #     # unpack current dataDf to X and y attributes
    #     self.unpackRecord()

    # def addRecord(self, newRecord):
    #     '''
    #     method adds new record (row) to dataDf
        
    #     Input:
    #     newRecord - dict with single key-val pair
    #         key - index value for new row in dataframe
    #         val - dict where keys are columns in dataframe
    #     '''
        
    #     # separate timestamp and record data   
    #     timestamp = list(newRecord.keys())[0] 
    #     newData = newRecord[timestamp] # data dict (could be nested)
    #     newDataFormatted = {} # flat data dict to add to df
        
    #     # check if there are new fields in record not found in self.dataDf
    #     dfColumns = self.dataDf.columns
        
    #     for field, fieldData in newData.items():

    #         # if fieldData is list, create multiple columns
    #         if type(fieldData) is list:

    #             # add field to self.dataMeta
    #             if field not in self.dataMeta.keys():
    #                 self.dataMeta[field] = {'tsIndexed': True, 'tsCount': len(fieldData)}

    #             # update self.dataMeta if tsIndexed changes
    #             if self.dataMeta[field].get('tsIndexed') is False:
    #                 logging.info(f'Existing field {field} now time indexed')
    #                 self.dataMeta[field] = {'tsIndexed': True, 'tsCount': len(fieldData)}
    #                 # rename column {field} to {field}_0 when time-index added
    #                 self.dataDf = self.dataDf.rename(columns={field : f'{field}_0'})

    #             # update tsCount if changed
    #             if len(fieldData) > self.dataMeta[field].get('tsCount'):
    #                 self.dataMeta[field]['tsCount'] = len(fieldData)

    #             # add each indexed data item to self.dataDf
    #             for ii, indexedFieldData in enumerate(fieldData):
    #                 # name if indexed field / colomn 
    #                 indexedFieldName = f'{field}_{ii}'
                    
    #                 # check if indexed field is column in df
    #                 if indexedFieldName not in self.dataDf.columns:
    #                     # add to df with None type as default value
    #                     self.dataDf[indexedFieldName] = None

    #                 # add current index data to dict-to-df
    #                 newDataFormatted[indexedFieldName] = indexedFieldData

    #         # if field is scalar, add directly
    #         else:
    #             # add field to self.dataMeta
    #             if field not in self.dataMeta.keys():
    #                 self.dataMeta[field] = {'tsIndexed': False}

    #             # check if indexed field is column in df
    #             if field not in self.dataDf.columns:
    #                 # if field missing, add to df with None type as default value
    #                 self.dataDf[field] = None

    #             # add field data directly to newDataFormatted
    #             newDataFormatted[field] = fieldData

    #     # write formatted data dict to new df row
    #     timestampIndex = pd.to_datetime(timestamp)
    #     self.dataDf.loc[timestampIndex] = newDataFormatted
        
    #     # unpack current dataDf to X and y attributes
    #     self.unpackRecord()
    
    # def saveRecord(self):
    #     '''
    #     method saves current self.dataDf to self.recordPath file as csv
    #     '''
        
    #     try:
    #         self.dataDf.to_csv(self.recordPath)
            
    #         self.backupDate = datetime.now()
            
    #     except Exception as e:
    #         traceback.print_exc()
    #         msg = f'Could not write training data to file {self.recordPath}'
    #         # logging.error(msg)
    #         # overwrite message with save error msg
    #         self.msg = msg 
    
    # def readRecord(self):
    #     '''
    #     method reads data from self.recordPath and unpacks to X and y attributes
    #     '''
    #     try:
    #         dataDf = pd.read_csv(self.recordPath, index_col = 0)
    #         dataDf.index = pd.to_datetime(dataDf.index) 

    #         self.dataDf = dataDf
            
    #         # unpack current dataDf to X and y attributes
    #         self.unpackRecord()
        
    #     except Exception as e:
    #         traceback.print_exc()
    #         msg = f'Could not read training data to file {self.recordPath}'
    #         # logging.error(msg)
    #         # overwrite message with load error msg
    #         self.msg = msg 
        
    # def checkRecordBackup(self):
    #     '''
    #     method checks if data backup should be perfomed based on self.backupInterval and last backupDate
    #     '''
        
    #     curTime = datetime.now()
        
    #     if (curTime - self.backupDate) > timedelta(seconds=self.backupInterval):
    #         logging.info('Saving training data to file')
    #         self.saveRecord()

    # def combineDataRecords(self,dataScada, dataWeather, ts=None, writeRecord=True):
    #     '''
    #     method combines data from SCADA and weather forecast sources
    #     and formats for storage in fcSelector record keeping
        
    #     inputs:
    #     ts - pandas timestamp used as index for record
    #     dataScada - string of json of scada record
    #     dataWeather - pandas df, ts indexed, cols contain ts inputs
    #     writeRecord - bool to write record to existing training dataset
    #             if False, returns new record as dict
        
    #     '''
        
    #     # init dataDict to store combine records
    #     dataDict = {}
        
    #     # unpack string json to dict
    #     dataScada = json.loads(dataScada)
        
    #     # loop through scada data and reformat
    #     dataScadaList = []
        
    #     # get keys for "cols" and "rows" of scada dict
    #     keysCols = list(dataScada.keys())
    #     keysRows = list(dataScada[keysCols[0]].keys())
        
    #     # loop through cols
    #     for rr in keysRows:
    #         # init new entry
    #         newEntry = {}
            
    #         for cc in keysCols:
    #             # extract data for current row/col
    #             newEntry[cc] = dataScada[cc][rr]
    #         # add to list of formatted scada data 
    #         dataScadaList.append(newEntry)
        
    #     # if scada data is valid, add to record
    #     valid = True # init data_valid flag to True
    #     for val in dataScadaList:
            
    #         # add scada data to update dict
    #         dataDict[val.get('name')] = val.get('value')
            
    #         # check if valid is set to false
    #         if not val.get('valid'):
    #             # trip flag to false if any scada data is false
    #             valid = False
                
    #     # add dataValid val to update dict
    #     dataDict['dataValid'] = valid
                
    #     # add weather data from each df col to dataDict
    #     for col in dataWeather.columns:
    #         dataDict[col] = list(dataWeather[col])
            
    #     # get ts from input arg, or weather df
    #     if ts is None:
    #         # if ts not provided in args, use first index from dataWeather df
    #         ts = dataWeather.first_valid_index()
            
    #     # package record as dict/pandas series
    #     newRecord = {
    #         ts: dataDict
    #     }
        
    #     if writeRecord:
    #         self.addRecord(newRecord)
    #     else:
    #         return newRecord

    # # data reshaping methods
    # def incrementTimestamp(self, ts, inc):
    #     '''
    #     method increments timestamp by inc hours. Used for reshaping wide ts dataframe columns

    #     Input:
    #     ts - timestamp str or pandas timestamp
    #     inc - number of hours to increment ts by
    #     '''
        
    #     # if ts is in str format, use self.dtFormat to format
    #     if type(ts) is str:
            
    #         ts = datetime.strptime(ts, self.dtFormat)
    #         ts = ts + timedelta(hours=inc)
    #         ts = ts.strftime(self.dtFormat)
            
    #     # if ts is in pd timestamp format
    #     if str(type(ts)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
            
    #         ts += pd.Timedelta(hours=inc)
            
    #     return ts

    # def reshapeTsData(self):
    #     '''
    #         method transforms self.dataDf containing hour-indexed forecast data columns
    #         into multiple separate rows for expanded training sets

    #         requires that self.dataMeta, which defines the contents and time-indexing of df columns
    #         to have been defined (either manually by user, or by appending new rows to self.dataDf)

    #         stores results in attr self.dataDfLong
    #     '''

    #     # if self.dataMeta has not been define, reshape cannot proceed, return without processing
    #     if self.dataMeta == {}:
    #         return None

    #     # extract dataframe from class instance
    #     df = self.dataDf.copy()
        
    #     # init reshaped df as empty
    #     df2 = None
        
    #     # move ts index to column / remove index
    #     df.reset_index(inplace=True)
    #     df = df.rename(columns = {'index':'timestamp'})
        
    #     # convert timestamp to pandas datetime
    #     df.timestamp = pd.to_datetime(df.timestamp)
        
    #     # get list of ts-indexed columns in df
    #     columnsScalar = []
    #     columnsIndexed = []
    #     hMax = 0
        
    #     for colName, colData in self.dataMeta.items():
    #         if colData.get('tsIndexed'):
    #             columnsIndexed.append(colName)
                
    #             # get highest tsCount (hMax)
    #             hMax = max(hMax, colData.get('tsCount'))
    #         else:
    #             columnsScalar.append(colName)
        
    #     # add index for forecast hour
    #     df['hIndex'] = None
    #     columnsScalar += ['timestamp']
        
    #     # extract non-ts data from df (df0)
    #     df0 = df.copy()[columnsScalar]
       
        
    #     # loop through hours in 0:hMax
    #     for hh in range(0, hMax):

    #         # get current h vals
    #         hCols = [f'{colName}_{hh}' for colName in columnsIndexed]
    #         hCols = ['timestamp', 'hIndex'] + hCols
            
    #         # create dict for renaming hCols
    #         hColRename = {f'{colName}_{hh}':colName for colName in columnsIndexed}

    #         # drop any hCols value that isn't a col in df (happens when ts-duration is not uniform)
    #         hCols = [val for val in hCols if val in df.columns]
            
    #         dfh = df.copy()[hCols]
    #         dfh['hIndex'] = hh
            
    #         dfh = dfh.rename(columns=hColRename)

    #         # update timestamp to shift hh hours later
    #         dfh['timestamp'] = dfh['timestamp'].apply(lambda y: self.incrementTimestamp(y, hh))

    #         # join to df on timestamp
    #         dfh = dfh.merge(df0, how='left', on='timestamp')

    #         # add to aggregate dataframe
    #         if df2 is None:
    #             # if first iteration, init df2 as current hour df (dfh)
    #             df2 = dfh.copy()
    #         else:
    #             df2 = pd.concat([df2, dfh])
                
    #     # set timestamp as index
    #     df2.set_index('timestamp', inplace=True)

    #     # if df2 has not been updated, default to df
    #     if df2 is None:
    #         df2 = self.dataDf
        
    #     self.dataDfLong = df2

    #     return