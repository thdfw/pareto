import sys
import os
import logging
import pandas as pd
import numpy as np
import json
import pickle
import traceback
import time
import datetime as dtm
import copy
import inspect
import pytz

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

# default config
DEFAULT_CONFIG = {
    'dataX': None,
    'dataY': None,
    'newRecord': {},
    'inputPath': None,
    'csvMetaData': {},
    'columnList': [],
    'scadaColumnList': None,
    'targetName': None,
    'unpackHourly': False,
    'useExternalTrainer': False,
    'isExternalTrainer': False,
    'externalTrainerPath': None,
    'backupData': None,        
    'backupPath': None,
    'backupInterval': None,
    'trainingInterval': 60*60*24*7, # default value 1 week
    'trainingIntervalBest': 60*60*24*2, # default value 2 days
    'trainingIntervalHyper': 60*60*24*7*2, # default value 2 week
    'trainingHour': 0, # default at midnight
    'predict_last': True, # predict based on last inputs
    'min_score': -1e3, # min score for valid forecaster
    'min_samples': 5*24,
    'inputProcessed': False,
    'fcParams': {},
    'newRecord': {},
    'removeModelsFromPickle': True,
    }
        
    
class ForecasterWrapperBase(eFMU):
    def __init__(self):
        self.input = {
            'forecaster-list': None, 
            'config': None,
            'weather-data': None,
            'scada-data': None,
            'data-timestamp': None,
            'input-data': None,
            'tz': None,
            'debug': None,
            'timeout': None,
            'last-trainall': None,
            'date-trainall': None,
        }  
        
        self.output = {
            'model-summary': None,
            'output-data': None,
            'duration': None,
            'date-trainall': None, # date for retraining for import-instance
            'last-trainall': None, # data of last training for external-instance
        }
        
        # set init flag for data initialization
        self.init = False

        # initialize data manager
        self.dataManager = None
        self.useExternalTrainer = False
        self.isExternalTrainer = False
        self.csvMetaData = {}
        
        # initialize selector and forecaster attributes
        self.trainingDate = None
        self.trainingDateBest = None
        self.framework = None
        self.bestModel = None
        
        # intialize dates for managing external loops
        self.retrainDateNext = None # dt for triggering next retraining
        self.retrainDateLast = None # dt for of most recent retraining trigger     
        self.extDataWritten = False # bool to indicate if training data has been written to ext file
        self._lastLoadedExt = None # date when external models were last loaded into wrapper
        
        # initialize selected forecaster prediction
        self.duration = None
        self.fcPrediction = None
        
        self.debug = None

        self.msg = ''
        self.log = ''
        
        self.dtFormat = '%Y-%m-%d %H:%M:%S' #'2018-07-21 01:00:00'
               
    def compute(self):

        # start compute duration timer
        self._computeSt = time.time()

        # initialize inputs
        self.compute_init()

        # train/load forecast models
        if self.valid:
            self.compute_train()
        else:
            return self.msg
        
        # manage data for wrappers using external forecasters
        self.manageExternalData()

        # generate new prediction
        if self.valid and self.isExternalTrainer is False:
            self.compute_predict()
        
        # write data to output dict
        self.compute_output()
        
        # Return status message
        return self.msg

    def compute_init(self):
        '''
        method initializes datasets when running compute
        '''

        # initialize msg to return
        self.msg = ''
        self.newPrediction = False
        self.valid = True        
       
        # extract all current inputs from self.inputs
        self.extractInputs()

        # # check for retraining if instance isExternalTrainer
        # if self.isExternalTrainer:
        #     self.checkRetraining()
        
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

    def compute_train(self):

        # if using external trainer, check if models should be loaded from pickle
        if self.useExternalTrainer:
            self.loadExternalModels()

            # define date for next retraining
            if self.trainingDate is not None:
                self.retrainDateNext = self.trainingDate + dtm.timedelta(seconds=self.trainingInterval)
                # align to training hour
                diffHours = self.retrainDateNext.hour - self.trainingHour
                if diffHours < 0:
                    self.retrainDateNext += dtm.timedelta(hours=-diffHours)
                elif diffHours > 0:
                    self.retrainDateNext += dtm.timedelta(hours=24-diffHours)
            return

        # otherwise, train models internally

        # confirm training data has been loaded
        if (self.trainingData['X'] is None) or (self.trainingData['y'] is None):
            self.msg += f'ERROR: Failed to generate training data.\n'
            return

        # check retraining status
        self.check_training_status()

        # if no retraining triggered, exit train step
        if not (self.doRetrainAll or self.doRetrainBest or self.doRetrainHyper):
            return
           
        # extract forecaster list from inputs
        self.fcList = json.loads(self.input.get('forecaster-list', '[]'))

        # if framework selection has not been made, run evaluate method
        if self.bestModel == None or self.doRetrainAll or (self.doRetrainBest and not self.isExternalTrainer):
        
            # if model hasn't been selected yet, run evaluate method
            try:
                #bestonly = True
                # reduce forecasters if bestonly
                fcList = self.fcList
                if self.bestModel and self.doRetrainBest and not self.doRetrainAll:
                    fcList = [fc for fc in self.fcList if fc['name'] == self.bestModel['name']]
                    self.logger.info('evaluating best model')
                else:
                    self.logger.info('evaluating all models')
                # initialize forecastor selection framwork
                self.framework = fcSelector.ForecasterFramework(params=self.fcParams,
                                                                data=self.trainingData, 
                                                                fcList=fcList,
                                                                debug=self.debug,
                                                                uid=self.logger_name)
                self.framework.evaluateAll()
                self.msg += self.framework.msg
                self.valid = self.valid and self.framework.validData
                
                if self.valid:
                    if self.framework.best['score_rmse'] > self.minScore:
                        self.bestModel = self.framework.best.to_dict()
                        self.trainingDateBest = self.now
                        if self.doRetrainAll  or not self.trainingDate:
                            self.trainingDate = self.now
                            
                        # if current instance self.isExternalTrainer, write forecaster data to external files
                        if self.isExternalTrainer:
                            self.logger.info('writing trained models to external file')
                            # delete models and predicitons to reduce pkl size
                            if self.removeModelsFromPickle:
                                del self.framework.fcData['model']
                                del self.framework.predictions
                                del self.framework.data
                            with open(self.externalTrainerPath, 'wb') as f:
                                pickle.dump(self.framework, f)
                    else:
                        self.msg += f'WARNING: training score of {self.framework.best["score_rmse"]} < {self.minScore}. Invalid training.'
                else:
                    self.msg += 'WARNING: training was not valid.'
                    
                 
            except Exception as e:
                self.msg += f'ERROR: Failed to train forecasters in fc list.\nForecaster message: {self.msg}\nError: {e}\n\n{traceback.format_exc()}\n'

        # define date for next retraining (Christoph: I don't think that's needed here)
        #if self.trainingDate is not None:
        #    self.retrainDateNext = self.trainingDate + dtm.timedelta(seconds=self.trainingInterval)

    def compute_predict(self):  
                
        # run prediction on fcData if passed
        try:
            # run prediction method of best
            if (self.inputData is not None) and self.bestModel:
                self.fcPrediction = pd.Series(self.bestModel['model'].predict(self.inputData).tolist())
                self.newPrediction = True
            elif self.predict_last and self.bestModel:
                X = self.trainingData['X'].iloc[-1:].copy()
                X.loc[X.index, 'obs_y'] = self.trainingData['y'].loc[X.index] # add current y to obs
                X = fcSelector.ForecasterFramework.add_index_features(None, X) # add index features to obs
                self.fcPrediction = pd.DataFrame(self.bestModel['model'].predict(X)).transpose()
                self.fcPrediction.index = [X.index[0] + pd.DateOffset(seconds=i*self.framework.stepsize) for i in range(len(self.fcPrediction))]
                self.fcPrediction.columns = ['y']
                self.newPrediction = True
            elif self.predict_last and not self.bestModel:
                self.msg += 'WARNING: Waiting to initialize forecaster.\n'
        except Exception as e:
            self.msg += f'ERROR: Failed to generate new prediction: {e}\n\n{traceback.format_exc()}\n'
            self.newPrediction = False
                     
    def compute_output(self):  

        # calc total compute duration
        self.duration = time.time()-self._computeSt
            
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
             
        
        self.output['duration'] = self.duration

        # write datetime of latest triggered retraining to output
        if self.useExternalTrainer:
            self.output['date-trainall'] = str(self.retrainDateLast)  
            
        # write date of most recent training if isExternalTrainer
        if self.isExternalTrainer:
            if self.framework is not None:
                self.output['last-trainall'] = str(self.framework.trainingDate) 
            else:
                self.output['last-trainall'] = None

    def check_training_status(self):
        '''
        method checks status of training, based on dates, intervals, and external triggers
        and determines whether retraining should be initiated during current run

        assigns the following attributes

        self.doRetrainAll: bool to retrain all models
        self.doRetrainBest: bool to retrain only best model
        self.doRetrainHyper: bool to initate retraining of model hyper parameters

        '''

        # init retraining flags to False
        self.doRetrainAll = False
        self.doRetrainBest = False
        self.doRetrainHyper  = False


        # check retraining if current instance is external trainer class
        if self.isExternalTrainer:

            # if no retrainData provided in inputs, exit iteration.
            if self.retrainDate is None:
                self.msg += f'WARNING: No training date provided. Suspending run.'
                self.valid = False
                return
            
            # check if retainData trigger is later than last retrain date
            # if last train date is later than trigger data, do not retrain
            # ensure that instance has a trainer framework with a previous trainingDate

            # set self.doRetrainAll to True
            # will be disabled for specific 2 specific cases:
            self.doRetrainAll = True

            # case 1: retraining trigger date is after current class self.now
            if self.retrainDate > self.now:
                self.doRetrainAll = False
                return

            # case 2: retraining has already occured for current retraing trigger date
            # 1 - a training framework exists
            # 2 - the training framework has a date for most recent training
            # 3 - that training date is after the input retraining trigger
            if self.framework is not None:
                if self.framework.trainingDate is not None:
                    # if previous training date is after input trigger, training is up to date, do nothing
                    if  self.retrainDate <= self.framework.trainingDate:
                        self.doRetrainAll = False
                        return
                    
            # if self.doRetrainAll is still True after checks, clear old trained model attributes
            if self.doRetrainAll:

                self.init = False
                self.dataManager = None
                self.csvMetaData = {}
                self.framework = None
                self.bestModel = None

        # if class instance is not external trainer, check for training using previous method
        else:
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

            # store training statuses in attributes
            self.doRetrainAll = retrainModels
            self.doRetrainBest = retrainBestOnly
    
    def parseDate(self, var):
    
        # avoid error on init (with -1 or "None")
        if isinstance(var, (int, float)):
            var = None
        elif isinstance(var, (str)):
            if var == 'None':
                var = None
                
        if var is not None:
            try:            
                # split off any trailing seconds digits
                var_split = var.split('.')[0]
                # convert read datetime str to datetime
                var = dtm.datetime.strptime(var_split, self.dtFormat)
            except Exception as e:
                self.msg += f'ERROR: Could not parse {var} to datetime: {e}\n'
                var = None
                
        return var
    
    
    def extractInputs(self):
        '''
        method extract all possible data fields from current value for self.inputs
        and stores them in corresponding class attributes
        '''
        
        # extract inputData for generating new prediction
        self.inputData = self.input.get('input-data', None)
        
        if 'time' in self.input:
            tz = pytz.timezone(self.input.get('tz'))
            self.now = dtm.datetime.fromtimestamp(self.input.get('time'), tz=tz).replace(tzinfo=None)
        else:
            self.logger.warning('using internal clock reference')
            self.now = pd.to_datetime(dtm.datetime.now()).to_pydatetime()
        
        # update logger
        self.uid = self.input.get('uid', 'root')
        self.logger_name = f'{self.uid}:{__name__}'
        self.logger = logging.getLogger(self.logger_name)
        self.debug = self.input.get('debug', False)
        if self.debug:
            if self.debug == 'info':
                self.logger.setLevel(logging.INFO)
            elif self.debug == 'warning':
                self.logger.setLevel(logging.WARNING)
            else:
                self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)
        
        # extract training data inputs from self.input
        self._config = copy.deepcopy(DEFAULT_CONFIG) # intialize with defaults
        self._config.update(self.input.get('config', {}))

        # initialize data flags
        self.dataValid = None
        self.dataMethod = None

        # extract attributes from inputs
        self.dataX = self._config['dataX']
        self.dataY = self._config['dataY']

        self.newRecord = self._config['newRecord']
        
        self.inputPath = self._config['inputPath']
        self.csvMetaData = self._config['csvMetaData']
        self.columnList = self._config['columnList']
        self.scadaColumnList = self._config['scadaColumnList']
        self.targetName = self._config['targetName']
        self.unpackHourly = self._config['unpackHourly']
        
        self.useExternalTrainer = self._config['useExternalTrainer']
        self.isExternalTrainer = self._config['isExternalTrainer']
        self.externalTrainerPath = self._config['externalTrainerPath']
        # self.externalFcListPath = self._config['externalFcListPath']
        
        self.backupData = self._config['backupData']        
        self.backupPath = self._config['backupPath']
        self.backupInterval = self._config['backupInterval']
        
        self.trainingInterval = self._config['trainingInterval']
        self.trainingIntervalBest = self._config['trainingIntervalBest']
        self.trainingIntervalHyper = self._config['trainingIntervalHyper']
        self.trainingHour = self._config['trainingHour']
        self.predict_last = self._config['predict_last']
        self.minScore = self._config['min_score']
        self.minSamples = self._config['min_samples']
        self.removeModelsFromPickle = self._config['removeModelsFromPickle']
        
        self._lastTrainAll = self.input.get('last-trainall', None) # get latest training datetime from input for externally trained models
        self._lastTrainAll = self.parseDate(self._lastTrainAll)
        self.retrainDate = self.input.get('date-trainall', None) # get datetime for retraining (for ext trainer instances only)
        self.retrainDate = self.parseDate(self.retrainDate)
        
        # check for flag that training data has already been processed on previous iteration
        self.inputProcessed = self._config['inputProcessed']
        self.fcParams = copy.deepcopy(self._config['fcParams'])
        
        self.fcParams['now'] = self.now.strftime(self.dtFormat)
        
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
        self.newRecord = self._config['newRecord']
        
        # check for new record fields in input
        self.dataScada = self.input.get('scada-data', None)
        self.dataWeather = self.input.get('weather-data', None)
        self.dataTs = self.input.get('data-timestamp', None)
        
        # check for update data in inputs
        self._scadaUpdate = (self.dataScada is not None and self.dataWeather is not None)
        self._simpleUpdate = self.newRecord is not None and bool(self.newRecord)
        self._doUpdate = self._scadaUpdate or self._simpleUpdate
        
        return
       
    def initData(self):
        '''
        method initializes training data based on items passed in 'config'.
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
        if self._config == {}:
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
            self.dataManager = fcDataMgmt.DataManager(params=self.mgmtParams,
                                                      debug=self.debug,
                                                      uid=self.logger_name)

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
        'config' input.
        
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
            #self.logger.debug('running backup check in wrapper')
            self.dataManager.checkRecordBackup()
            
        
        return
            
    def manageExternalData(self):
        '''
        method manages writing data to external files. 
        
        used only when self.useExternalTrainer is true

        '''
        
        # check if self.useExternalTrainer
        if not self.useExternalTrainer:
            return
        
        
        # check if adaquate data has been stored
        nSamples = self.dataManager.dataDf.shape[0]
        if not (nSamples > self.minSamples):
            # not enough data stored, do nothing
            # print(f'not enough data to export training dataset: {nSamples}')
            return
        
        # check if no previous forecasters have been loaded
        # if self.retrainDateLast is None:
            
        # if retainDataNext is also None, data has never been written, so proceed
        if self.retrainDateNext is None:
            # self retrain data to now, to trigger dump
            self.retrainDateNext = self.now
            # print('setting initial retraining data')
        elif self.retrainDateNext == self.retrainDateLast:
            # if self retrainDateNext has already been defined and written to ouputs, exit
            # print('training data set, waiting for trained forecasters')
            return
            
        
        # for expired or previously-none retrainDateNext, dump data to file
        if self.retrainDateNext <= self.now:
            # print(f'exporting training data samples: {nSamples}')

            try:
                # print('retraining-date should now be updated')
                # store current triggered training data in retrainDateLast (to be written to outputs)
                self.retrainDateLast = self.retrainDateNext
                # write trainging data to external file
                self.dataManager.backupData = True
                self.dataManager.saveRecord()
                
            except Exception as e:
                self.msg += f'ERROR: Could not save training data to external file.\nError: {e}\n\n{traceback.format_exc()}\n'
                # print('retraining-date update failed')

        return

    def loadExternalModels(self):
        '''
        method for checking if externally trained models should be loaded into class instance

        should only be used if self.useExternalTrainer is True

        populates the following class attributes
        framework: model selection framework with trained models and training data
        bestModel: the model selected by the above framework
        traingingDate: the date when model where last trained
        _lastLoadedExt: the date when external models where loaded into this class instace

        '''

        # confirm that useExternalTrainer is set to True
        if not self.useExternalTrainer:
            return

        try:
            # check if last-trainall is provided/valid
            if self._lastTrainAll is not None:
                if self._lastTrainAll <= self.now:
                    
                    # only load forecasters if not previously loaded
                    if self._lastTrainAll != self._lastLoadedExt:
            
                        # extract externalTrainer (fcSelector instance) and fcList from external files
                        self.logger.info('reading trained models from external file')
                        with open(self.externalTrainerPath, 'rb') as fExt:
                            
                            framework = pickle.load(fExt)

                            self.externalTrainer = framework
                            self.framework = self.externalTrainer
                        
                        
                        # self.fcList = json.loads(self.externalFcListPath )
                        self.fcList = self.externalTrainer.fcList
                        self.bestModel = self.externalTrainer.best.to_dict()
                        
                        # self.trainingData = self.externalTrainer.data
                    
                        # get training data from external trainer interal data
                        self.trainingDate = self.externalTrainer.trainingDate
                        
                        # set date for most recent forecaster loading
                        self._lastLoadedExt = self._lastTrainAll


            elif self.bestModel is None:
                # if loaded framework does not include a bestModel, add error to msg
                self.valid = False
                self.msg += 'WARNING: Externally trained model not yet avialable. No prediction generated.'

        except Exception as e:
            self.valid = False
            self.msg += f'ERROR: Failed to open external forecast model\nError: {e}\n\n{traceback.format_exc()}\n'


class ForecasterWrapper(ForecasterWrapperBase):
    def __init__(self):
        ForecasterWrapperBase.__init__(self)
        
        self.input = {
            'forecaster-list': None, 
            'config': None,
            'weather-data': None,
            'scada-data': None,
            'data-timestamp': None,
            'input-data': None,
            'tz': None,
            'debug': None,
            'timeout': None,
            'last-trainall': None,
        }  
        
        self.output = {
            'model-summary': None,
            'output-data': None,
            'duration': None,
            'date-trainall': None, # date for retraining for import-instance
        }
        
        
class ForecasterTrainer(ForecasterWrapperBase):
    def __init__(self):
        ForecasterWrapperBase.__init__(self)
        
        self.input = {
            'forecaster-list': None, 
            'config': None,
            'tz': None,
            'debug': None,
            'timeout': None,
            'date-trainall': None,
        }  
        
        self.output = {
            'model-summary': None,          
            'duration': None,
            'last-trainall': None, # data of last training for external-instance
        }


if __name__ == "__main__":
    
    pass

    
