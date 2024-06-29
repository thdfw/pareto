import os
import sys
import time
import json
import logging
import warnings
import pandas as pd
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import traceback

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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



class DataManager():
    '''
    This class 
    methods:
    __init__: Initializes the framework with a list of models to 
              evaulate, and a training dataset to use for model
              training and testing.
    
    
              
    '''
    
    def __init__(self, params={}, debug=False, uid='root'):
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
        
        self.debug = debug
        self.uid = uid
        
        self.updateParams(params)

        self.data = None
        self.X = None
        self.y = None
        
        self.dataDf = None
        self.dataDfLong = None
        self.backupDate = None
        # self.dataMeta = {} # dict of meta-data describing training data fields

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
        
        
        if self.data:
            self.updateData(self.data)
        else:
            self.initRecord()  
            
    def updateParams(self, params={}):
        
        self.params = params
        
        # parse parameter
        self.colList = self.params.get('colList', [])
        self.scadaColumnList = self.params.get('scadaColumnList', [])
        self.targetColName = self.params.get('targetColName', None)
        self.backupData = self.params.get('backupData', True)
        self.backupPath = self.params.get('backupPath', './data.csv')
        self.backupInterval = self.params.get('backupInterval', 7*24*60*60)
        self.unpackHourly = bool(self.params.get('unpackHourly', True))
        self.now = self.params.get('now', datetime.now())
        self.dataMeta = self.params.get('csvMetaData', {})


        # make sure backupInterval is int
        try:
            self.backupInterval = int(self.backupInterval)
        except:
            # if int conversion fails, default to 1 week
            self.backupInterval = 7*24*60*60
        

    # data management methods
    def unpackRecord(self):
        '''
        method unpacks record df into X and y using self.targetColName

        if unpackHourly is enabled, method will reshape training data to create
        more observations for forecasted input (e.g. hourly temperature forecasts)
        '''
        
        # init data attr to empty
        self.data = {}
        self.X = None
        self.y = None
        
        # if no data stored, do nothing
        if self.dataDf is None:
            return
        
        if self.dataDf.shape[0] == 0:
            return

        # check if unpackHourly is enabled
        if self.unpackHourly:

            # if no column metadata provided, do nothing
            if self.dataMeta == {}:
                
                # can't unpack hourly data if these is no metaData
                self.msg += 'WARNING: Cannot unpack hourly data without metadata \n'
                self.unpackHourly = False
                
                dataDf = self.dataDf.copy()
                
            else:

                # use reshapeTsData method to unpack hourly columns
                self.reshapeTsData()
                dataDf = self.dataDfLong.copy()

        # otherwise, use self.dataDf.columns directly
        else:
            dataDf = self.dataDf.copy()

        #  drop rows with NaN values
        #dataDf.dropna(inplace=True)
        
        # sort data
        dataDf = dataDf.sort_index()

        # get list of all columns
        dfColsList = list(dataDf.columns)

        # check if targetCol is currently in df
        if self.targetColName is None or self.targetColName not in dfColsList:
            # if target is not in list of column, use last column from df
            msg = f"Target column {self.targetColName} not found in data.\n"
            self.logger.warning(msg)
            self.msg += msg

            # if no target provided, set data to None so that forecaster cannot be run
            self.data = {
                'X': None,
                'y': None
            }

            self.X = None
            self.y = None

            return 

        # if target column is defined, proceed with data unpacking
        targetColName = self.targetColName

        # divide data based on current targetColName
        dfColsList.remove(targetColName)
        columnsX = dfColsList
        columnsY = targetColName

        self.data = {
            'X': dataDf[columnsX],
            'y': dataDf[columnsY]
        }

        self.X = dataDf[columnsX]
        self.y = dataDf[columnsY]

        # self.validateData()

    def initRecord(self):
        '''
        method initializes record dataframe. Creates empty dataframe
        '''
        
        colList = self.colList

        # add columns to self.dataMeta
        if colList is not None:
            for col in colList:
                self.dataMeta[col] = {}

        # remove timestamp from list, which is used as index
        if 'timestamp' in colList:
            colList.remove('timestamp')

        # initialize df
        df = pd.DataFrame(columns = colList)

        self.dataDf = df

        # unpack current dataDf to X and y attributes
        self.unpackRecord()

    def addRecord(self, newRecord):
        '''
        method adds new record (row) to dataDf
        
        Input:
        newRecord - dict with single key-val pair
            key - index value for new row in dataframe
            val - dict where keys are columns in dataframe
        '''
        
        # separate timestamp and record data   
        timestamp = list(newRecord.keys())[0] 
        newData = newRecord[timestamp] # data dict (could be nested)
        newDataFormatted = {} # flat data dict to add to df
        
        # check if there are new fields in record not found in self.dataDf
        dfColumns = self.dataDf.columns
        
        for field, fieldData in newData.items():

            # if fieldData is list, create multiple columns
            if type(fieldData) is list:

                # add field to self.dataMeta
                if field not in self.dataMeta.keys():
                    self.dataMeta[field] = {'tsIndexed': True, 'tsCount': len(fieldData)}

                # update self.dataMeta if tsIndexed changes
                if self.dataMeta[field].get('tsIndexed') is False:
                    self.logger.info(f'Existing field {field} now time indexed')
                    self.dataMeta[field] = {'tsIndexed': True, 'tsCount': len(fieldData)}
                    # rename column {field} to {field}_0 when time-index added
                    self.dataDf = self.dataDf.rename(columns={field : f'{field}_0'})

                # update tsCount if changed
                if len(fieldData) > self.dataMeta[field].get('tsCount'):
                    self.dataMeta[field]['tsCount'] = len(fieldData)

                # add each indexed data item to self.dataDf
                for ii, indexedFieldData in enumerate(fieldData):
                    # name if indexed field / colomn 
                    indexedFieldName = f'{field}_{ii}'
                    
                    # check if indexed field is column in df
                    if indexedFieldName not in self.dataDf.columns:
                        # add to df with None type as default value
                        self.dataDf[indexedFieldName] = None

                    # add current index data to dict-to-df
                    newDataFormatted[indexedFieldName] = indexedFieldData

            # if field is scalar, add directly
            else:
                # add field to self.dataMeta
                if field not in self.dataMeta.keys():
                    self.dataMeta[field] = {'tsIndexed': False}

                # check if indexed field is column in df
                if field not in self.dataDf.columns:
                    # if field missing, add to df with None type as default value
                    self.dataDf[field] = None

                # add field data directly to newDataFormatted
                newDataFormatted[field] = fieldData

        # write formatted data dict to new df row
        timestampIndex = pd.to_datetime(timestamp)
        self.dataDf.loc[timestampIndex] = newDataFormatted
        
        # unpack current dataDf to X and y attributes
        self.unpackRecord()
    
    def saveRecord(self):
        '''
        method saves current self.dataDf to self.backupPath file as csv
        '''
        
        # if self.backupData is not set to True, do not save data
        if not self.backupData:
            return
            
        
        try:
            self.dataDf.to_csv(self.backupPath)
            
            self.backupDate = self.now
            
        except Exception as e:
            msg = f'Could not write training data to file {self.backupPath}: {e}\n\n{traceback.format_exc()}\n'
            self.logger.error(msg)
            # overwrite message with save error msg
            self.msg += msg 
    
    def readRecord(self, inputPath=None, dataMeta=None):
        '''
        method reads data from self.backupPath and unpacks to X and y attributes
        '''
        try:

            # if no inputPath provided, read from record path
            #if inputPath is None:
            #    inputPath = self.backupPath
            
            # update metaData if provided, otherwise use existing
            if dataMeta is not None and dataMeta != {}:
                self.dataMeta = dataMeta

            dataDf = pd.read_csv(inputPath, index_col = 0)
            dataDf.index = pd.to_datetime(dataDf.index) 

            self.dataDf = dataDf
            
            # unpack current dataDf to X and y attributes
            self.unpackRecord()
        
        except Exception as e:
            msg = f'Could not read training data from file {inputPath}: {e}\n\n{traceback.format_exc()}\n'
            self.logger.error(msg)
            # overwrite message with load error msg
            self.msg += msg 
        
    def checkRecordBackup(self):
        '''
        method checks if data backup should be perfomed based on self.backupInterval and last backupDate
        '''
        
        #self.logger.info('checking backup interval')
        
        curTime = self.now
        
        # initialize backup date with now
        if not self.backupDate:
            self.backupDate = curTime
        
        tDelta = curTime - self.backupDate
        
        if tDelta > timedelta(seconds=self.backupInterval):
            self.logger.info('saving training data to file')
            self.saveRecord()
        else:
            self.logger.debug(f'backup interval not met {tDelta} - {self.backupInterval}. No backup executed')

    def combineDataRecords(self, dataScada, dataWeather, ts=None, writeRecord=True):
        '''
        method combines data from SCADA and weather forecast sources
        and formats for storage in fcSelector record keeping
        
        inputs:
        ts - pandas timestamp used as index for record
        dataScada - string of json of scada record
        dataWeather - pandas df, ts indexed, cols contain ts inputs
        writeRecord - bool to write record to existing training dataset
                if False, returns new record as dict
        
        '''
        
        # init dataDict to store combine records
        dataDict = {}
        
        # unpack string json to dict
        dataScada = json.loads(dataScada)
        dataWeather = pd.read_json(dataWeather)
        
        # loop through scada data and reformat
        dataScadaList = []
        
        # get keys for "cols" and "rows" of scada dict
        keysCols = list(dataScada.keys())
        keysRows = list(dataScada[keysCols[0]].keys())
        
        # filter scada data
        #if self.scadaColumnList:
        #    keysCols = self.scadaColumnList + [self.targetColName]
        
        # loop through cols
        for rr in keysRows:
            # init new entry
            newEntry = {}
            
            for cc in keysCols:
                # extract data for current row/col
                newEntry[cc] = dataScada[cc][rr]
            # add to list of formatted scada data 
            dataScadaList.append(newEntry)
        
        # if scada data is valid, add to record
        valid = True # init data_valid flag to True
        for val in dataScadaList:
            
            # filter scada data
            if self.scadaColumnList:
                if val.get('name') not in (self.scadaColumnList + [self.targetColName]):
                    continue
                
            # add scada data to update dict
            dataDict[val.get('name')] = val.get('value')
            
            # check if valid is set to false
            if not val.get('valid'):
                # trip flag to false if any scada data is false
                valid = False
                
        # add dataValid val to update dict
        dataDict['dataValid'] = valid
                
        # add weather data from each df col to dataDict
        for col in dataWeather.columns:
            dataDict[col] = list(dataWeather[col])
            
        # get ts from input arg, or weather df
        if ts is None:
            # if ts not provided in args, use first index from dataWeather df
            ts = dataWeather.first_valid_index()
            
        # package record as dict/pandas series
        newRecord = {
            ts: dataDict
        }
        
        if writeRecord:
            self.addRecord(newRecord)
        else:
            return newRecord

    # data reshaping methods
    def incrementTimestamp(self, ts, inc):
        '''
        method increments timestamp by inc hours. Used for reshaping wide ts dataframe columns

        Input:
        ts - timestamp str or pandas timestamp
        inc - number of hours to increment ts by
        '''
        
        # if ts is in str format, use self.dtFormat to format
        if type(ts) is str:
            
            ts = datetime.strptime(ts, self.dtFormat)
            ts = ts + timedelta(hours=inc)
            ts = ts.strftime(self.dtFormat)
            
        # if ts is in pd timestamp format
        if str(type(ts)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
            
            ts += pd.Timedelta(hours=inc)
            
        return ts

    def reshapeTsData(self):
        '''
            method transforms self.dataDf containing hour-indexed forecast data columns
            into multiple separate rows for expanded training sets

            requires that self.dataMeta, which defines the contents and time-indexing of df columns
            to have been defined (either manually by user, or by appending new rows to self.dataDf)

            stores results in attr self.dataDfLong
        '''

        # if self.dataMeta has not been define, reshape cannot proceed, return without processing
        if self.dataMeta == {}:
            return None

        # extract dataframe from class instance
        df = self.dataDf.copy()
        
        # init reshaped df as empty
        df2 = None
        
        # move ts index to column / remove index
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'timestamp'})
        
        # convert timestamp to pandas datetime
        df.timestamp = pd.to_datetime(df.timestamp)
        
        # get list of ts-indexed columns in df
        columnsScalar = []
        columnsIndexed = []
        hMax = 0
        
        for colName, colData in self.dataMeta.items():
            if colData.get('tsIndexed'):
                columnsIndexed.append(colName)
                
                # get highest tsCount (hMax)
                hMax = max(hMax, colData.get('tsCount'))
            else:
                columnsScalar.append(colName)
        
        # add index for forecast hour
        df['hIndex'] = None
        columnsScalar += ['timestamp']
        
        # extract non-ts data from df (df0)
        df0 = df.copy()[columnsScalar]
       
        
        # loop through hours in 0:hMax
        for hh in range(0, hMax):

            # get current h vals
            hCols = [f'{colName}_{hh}' for colName in columnsIndexed]
            hCols = ['timestamp', 'hIndex'] + hCols
            
            # create dict for renaming hCols
            hColRename = {f'{colName}_{hh}':colName for colName in columnsIndexed}

            # drop any hCols value that isn't a col in df (happens when ts-duration is not uniform)
            hCols = [val for val in hCols if val in df.columns]
            
            dfh = df.copy()[hCols]
            dfh['hIndex'] = hh
            
            dfh = dfh.rename(columns=hColRename)

            # update timestamp to shift hh hours later
            dfh['timestamp'] = dfh['timestamp'].apply(lambda y: self.incrementTimestamp(y, hh))

            # join to df on timestamp
            dfh = dfh.merge(df0, how='left', on='timestamp')

            # add to aggregate dataframe
            if df2 is None:
                # if first iteration, init df2 as current hour df (dfh)
                df2 = dfh.copy()
            else:
                df2 = pd.concat([df2, dfh])
                
        # set timestamp as index
        df2.set_index('timestamp', inplace=True)

        # if df2 has not been updated, default to df
        if df2 is None:
            df2 = self.dataDf
        
        self.dataDfLong = df2

        return