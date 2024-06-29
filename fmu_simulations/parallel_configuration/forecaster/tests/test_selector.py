import unittest
import os
import sys

import pandas as pd

# Append parent directory to import DOPER
sys.path.append('../src')

import fcLib
from fcSelector import ForecasterFramework, default_params


class TestSelector(unittest.TestCase):
    '''
    
    unit tests for running basic Forecaster Selector class  
    
    '''
    
    # set initial setUpComplete flag to false, so training is run on first test
    setupComplete = False
    # set number of forecaster candidates to include in testng
    nCadidates = 3
    
    
    def setUp(self):
        '''
        setUp method is run before each test.
        Only one forecaster training is needed for all tests, 
        so if self.setupComplete is True, the training step is skipped

        '''
        
        if not hasattr(self, 'setupComplete'):
            print('Initializing test: training forecasters')
            self.runTraining()
        else:
            if self.setupComplete is False:
                print('Initializing test: training forecasters')
                self.runTraining()
            else:
                print('Test training has been completed')
        
        
    def runTraining(self):
        '''
        run training and store outputs as attributes of the class

        '''

        # create a forecaster library using list of models and parameters
        library = fcLib.forecasters(fcLib.forecaster_list, fcLib.models)
        fcList = fcLib.forecaster_list

        # reduce fcList to max candidates
        fcList = fcList[0:self.nCadidates]

        # load training data
        cwd = os.getcwd()
        folder = os.path.join(cwd, '..', 'resources', 'data')

        data = pd.read_csv(os.path.join(folder, 'forecaster_example_data.csv'), index_col = 0)
        data.index = pd.to_datetime(data.index)

        # Split the data into X and y
        X_columns = [col for col in data.columns if not 'Ppv_forecast' in col]
        y_columns = 'Ppv_forecast_1'

        X = data[X_columns]
        y = data[y_columns]

        # package data for framework
        data_eval = {
            'X': X,
            'y': y
        }

        # instantiate selector object, train and test
        params = default_params.copy()
        params['train_method'] = 'daily_split'
        params['min_days'] = 6

        a = ForecasterFramework(params=params, data=data_eval, fcList=fcList)

        a.evaluateAll(parallel=False)
        
        
        # package model results into unit test class attributes
        self.__class__.predictions = a.predictions
        self.__class__.bestModel = a.bestModel
        self.__class__.bestModelName = a.bestModelName
        self.__class__.bestScore = a.bestScore
        self.__class__.fcData = a.fcData
        
        # change setupComplete to True
        self.__class__.setupComplete = True
   
    
    # check that setup optimization has created expected result objects
    def test_exists_predictions(self):
        self.assertTrue(hasattr(self, 'predictions'), msg='predictions does not exist')
        
    def test_exists_bestModel(self):
        self.assertTrue(hasattr(self, 'bestModel'), msg='bestModel does not exist')
        
    def test_exists_bestModelName(self):
        self.assertTrue(hasattr(self, 'bestModelName'), msg='bestModelName does not exist')
        
    def test_exists_bestScore(self):
        self.assertTrue(hasattr(self, 'bestScore'), msg='bestScore does not exist')
        
    def test_exists_fcData(self):
        self.assertTrue(hasattr(self, 'fcData'), msg='fcData does not exist')

    # check that returned attributes are correct type
    def test_type_fcData(self):
        self.assertEqual(str(type(self.fcData)), "<class 'pandas.core.frame.DataFrame'>", msg='fcData incorrect type') 

    def test_type_bestScore(self):
        self.assertEqual(str(type(self.bestScore)), "<class 'numpy.float64'>", msg='bestScore incorrect type') 
        
    def test_type_predictions(self):
        self.assertEqual(type(self.predictions), list, msg='predictions incorrect type') 
    
    # check that returned attributes have correct number of elements
    def test_size_fcData(self):
        self.assertEqual(self.fcData.shape[0], self.nCadidates, msg='fcData incorrect number of results')    
    
    def test_size_predictions(self):
        self.assertEqual(len(self.predictions), self.nCadidates, msg='predictions incorrect number of results') 

if __name__ == '__main__':
    unittest.main()