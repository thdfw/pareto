import unittest
import os
import sys

from datetime import datetime, timedelta

# Append parent directory to import DOPER
sys.path.append('../src')

from fcSelector import ForecasterFramework, default_params


class TestReshape(unittest.TestCase):
    '''
    
    unit tests for running basic data management methods in selector class
    
    '''
    
    # set initial setUpComplete flag to false, so data operatations is run on first test
    setupComplete = False
    
    
    def setUp(self):
        '''
        setUp method is run before each test.
        Only one data processing operation is needed for all tests, 
        so if self.setupComplete is True, the processing step is skipped

        '''
        
        if not hasattr(self, 'setupComplete'):
            print('Initializing test: processing training data')
            self.runData()
        else:
            if self.setupComplete is False:
                print('Initializing test: processing training data')
                self.runData()
            else:
                print('Data processing has been completed')
        
        
    def runData(self):
        '''
        run data processing and store outputs as attributes of the class

        '''
        recordPath = './training_data.csv'

        # set params for data record
        parameter = {
            'targetColName': 't',
            'unpackHourly': True
        }

        # create an empty forecaster framework
        a = ForecasterFramework(params = parameter)

        # define example training data to add
        # including time-index input fields
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

        self.__class__.nFields = 6
        self.__class__.nRecords = len(newRecords) # should be 7
        self.__class__.nHours = 5 # max number of hourly observations

        self.__class__.expShapeX = (25,6)
        self.__class__.expShapeY = 25

        # add records to framework a
        for newRecord in newRecords:

            a.addRecord(newRecord)

        a.reshapeTsData()

        
        # package model results into unit test class attributes
        self.__class__.dataMeta = a.dataMeta
        self.__class__.dataDf = a.dataDf
        self.__class__.dataDfLong = a.dataDfLong

        self.__class__.metaCount = len(a.dataMeta)
        self.__class__.dfShapeWide = a.dataDf.shape
        self.__class__.dfShapeLong = a.dataDfLong.shape

        self.__class__.dimsX = a.X.shape
        self.__class__.dimsY = a.y.shape[0]
        
        # change setupComplete to True
        self.__class__.setupComplete = True
   
    
    # check that setup optimization has created expected result objects
    def test_exists_dataMeta(self):
        self.assertTrue(hasattr(self, 'dataMeta'), msg='Column meta data does not exist')

    def test_exists_dataDf(self):
        self.assertTrue(hasattr(self, 'dataDf'), msg='Training dataframe does not exist')
        
    def test_exists_dataDfLong(self):
        self.assertTrue(hasattr(self, 'dataDfLong'), msg='Training dataframe (long) does not exist')

    
    # check that returned training datasets have correct number of elements
    def test_size_dataMeta(self):
        self.assertEqual(self.metaCount, self.nFields, msg='Column meta data incorrect count') 

    def test_size_dataX(self):
        self.assertEqual(self.dimsX, self.expShapeX, msg='Reshaped X data incorrect shape') 

    def test_size_dataY(self):
        self.assertEqual(self.dimsY, self.expShapeY, msg='Reshaped y data (a) incorrect size')  

    

    
    

if __name__ == '__main__':
    unittest.main()