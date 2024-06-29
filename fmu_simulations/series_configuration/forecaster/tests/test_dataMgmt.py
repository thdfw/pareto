import unittest
import os
import sys

from datetime import datetime, timedelta

# Append parent directory to import DOPER
sys.path.append('../src')

from fcSelector import ForecasterFramework, default_params


class TestSelector(unittest.TestCase):
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
            'colList': ['c1', 'c2', 'c3'],
            'recordPath': recordPath,
            'backupInterval': timedelta(seconds=5),
            'unpackHourly': False
        }

        # create 2 empty forecaster frameworks
        a = ForecasterFramework(params = parameter)
        b = ForecasterFramework(params = parameter)

        # define example training data to add
        newRecords = [
            {
                '2018-07-21 00:00:00': {
                    'c1': 11,
                    'c2': 22,
                    'c3': 1,
                    't': 101
                }
            },
            {
                '2018-07-21 01:00:00': {
                    'c1': 13,
                    'c2': 25,
                    'c3': 2,
                    't': 103
                }
            },
            {
                '2018-07-21 02:00:00': {
                    'c1': 15,
                    'c2': 35,
                    'c3': 4,
                    't': 112
                }
            },
            {
                '2018-07-21 03:00:00': {
                    'c1': 16,
                    'c2': 44,
                    'c3': 5,
                    't': 124
                }
            },
            {
                '2018-07-21 04:00:00': {
                    'c1': 17,
                    'c2': 49,
                    'c3': 6,
                    't': 144
                }
            }
        ]

        self.__class__.nFields = 3
        self.__class__.nRecords = len(newRecords)

        # add records to framework a
        for newRecord in newRecords:

            a.addRecord(newRecord)

        # save record to file
        a.saveRecord()

        # read stored data into framework b
        b.readRecord()

        # delete external data record file 
        if os.path.exists(recordPath):
            os.remove(recordPath)
        
        # package model results into unit test class attributes
        self.__class__.aX = a.X
        self.__class__.aY = a.y
        self.__class__.bX = b.X
        self.__class__.bY = b.y
        
        # change setupComplete to True
        self.__class__.setupComplete = True
   
    
    # check that setup optimization has created expected result objects
    def test_exists_aDataX(self):
        self.assertTrue(hasattr(self, 'aX'), msg='X data (a) does not exist')

    def test_exists_aDataY(self):
        self.assertTrue(hasattr(self, 'aY'), msg='y data (a) does not exist')
        
    def test_exists_bDataX(self):
        self.assertTrue(hasattr(self, 'bX'), msg='X data (b) does not exist')

    def test_exists_bDataY(self):
        self.assertTrue(hasattr(self, 'bY'), msg='y data (b) does not exist')

    # check that returned attributes are correct type
    def test_type_aDataX(self):
        self.assertEqual(str(type(self.aX)), "<class 'pandas.core.frame.DataFrame'>", msg='X data (a) incorrect type')
    
    def test_type_aDataY(self):
        self.assertEqual(str(type(self.aY)), "<class 'pandas.core.series.Series'>", msg='y data (a) incorrect type')

    def test_type_bDataX(self):
        self.assertEqual(str(type(self.bX)), "<class 'pandas.core.frame.DataFrame'>", msg='X data (b) incorrect type')
    
    def test_type_bDataY(self):
        self.assertEqual(str(type(self.bY)), "<class 'pandas.core.series.Series'>", msg='y data (b) incorrect type')

    
    # check that returned training datasets have correct number of elements
    def test_size_aDataX(self):
        self.assertEqual(self.aX.shape, (self.nRecords, self.nFields), msg='X data (a) incorrect shape') 

    def test_size_aDataY(self):
        self.assertEqual(self.aY.size, self.nRecords, msg='y data (a) incorrect size')  

    def test_size_bDataX(self):
        self.assertEqual(self.bX.shape, (self.nRecords, self.nFields), msg='X data (b) incorrect shape') 

    def test_size_bDataY(self):
        self.assertEqual(self.bY.size, self.nRecords, msg='y data (b) incorrect size')  
    
    

if __name__ == '__main__':
    unittest.main()