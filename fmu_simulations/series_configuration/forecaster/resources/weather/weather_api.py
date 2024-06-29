# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:19:52 2015

@author: ryan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:32:46 2015

@author: ryan
"""
import datetime
import time
import urllib2
import json
import csv

def get_weatherHistory(api_key, latitude, longitude, start, end):
    '''this function returns a csv file with weather conditions within the specified start and end times
    '''
    timeInc = datetime.timedelta(days = 1)
    currentStamp = int(start)
    endStamp = int(end)
    currentDate = datetime.datetime.fromtimestamp(currentStamp)
    fileToWrite = open(start+'to'+end+'.csv', 'wt')
    writer = csv.writer(fileToWrite)
    writer.writerow(('timestamp', 'summary', 'icon', 'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature',
                       'dewPoint', 'humidity', 'windSpeed', 'windBearing', 'visibility', 'cloudCover', 'pressure'))
    while currentStamp <= endStamp:
        timestamp = str(currentStamp)
        # Collect past weather condition using Dark Sky Forecast API for specified location and timestamp
        # in SI units, using specified API key

        api_str = 'https://api.forecast.io/forecast/' + api_key + '/' + latitude + ',' + longitude + ','+ timestamp + '?units=si'
        req = urllib2.Request(api_str)
        f = urllib2.urlopen(req)
        json_string = f.read()
        parsed_json = json.loads(json_string)

        #history = parsed_json['currently']  # to get the current condition only
        hourly_ob = parsed_json['hourly']  # to get the hourly conditions
        #daily_ob = parsed_json['daily']  # to get the daily condition
        localTime = float(timestamp)
        localTime = time.localtime(localTime)

        # save the whole json file
        #json_out = open(timestamp+'.json', 'w')
        #json.dump(json_string, json_out, indent = None )
        #json_out.close()

        # save only the hourly data of the json file
        json_hourly = open('hourly'+timestamp+'.json', 'w')
        json.dump(hourly_ob, json_hourly, indent = None)
        json_hourly.close()

        # create a csv file from the hourly json file
        condition_str = ['time', 'summary', 'icon', 'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature',
                       'dewPoint', 'humidity', 'windSpeed', 'windBearing', 'visibility', 'cloudCover', 'pressure']
        condition_var = ['timestamp', 'summary', 'icon', 'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature',
                       'dewPoint', 'humidity', 'windSpeed', 'windBearing', 'visibility', 'cloudCover', 'pressure']
        for i in range(0, len(hourly_ob['data'])):
            for ii in range(0,14):
                if condition_str[ii] in hourly_ob['data'][i]:
                    condition_var[ii] = str(hourly_ob['data'][i][condition_str[ii]])
                else:
                    condition_var[ii] = ''
            writer.writerow(condition_var)

        # iterate for next request using the next timestamp
        currentDate = currentDate + timeInc
        currentDate = datetime.datetime.timetuple(currentDate)
        currentStamp = int(time.mktime(currentDate))
        currentDate = datetime.datetime.fromtimestamp(currentStamp)
        date = time.strftime('%Y-%m-%d' , localTime)
        print '24H weather data for ' + date + 'was saved as ' + timestamp +'.json'
        f.close()
        #time.sleep(2)  # time delay if needed for internet response
    else:
        print 'Finished saving weather history!'
        print 'Check working folder for filename: ' +start+'to'+end+'.csv'

    fileToWrite.close()
    return

###################### Main #####################################################################################
# Uncomment the following line to download historical weather data from the web
import os
intfile_path = '/Users/nxd/Desktop/Work Files/MG Controller/forecasters/resources/weather/api_data'
os.chdir(intfile_path)

apiKey = 'API KEY GOES HERE'

#get_weatherHistory(apiKey, '33.941794','-118.408519', '1339484400', '1425110400')
#get_weatherHistory(apiKey, '33.941794','-118.408519', '1425196800', '1430722800')
# get_weatherHistory(apiKey, '33.941794','-118.408519', '1339398000', '1339484400')
# get_weatherHistory(apiKey, '37.71512771387348', '-121.90834387103284', '1601535600', '1627801200')

# 2019
#get_weatherHistory(apiKey, '37.71512771387348', '-121.90834387103284', '1546326000', '1577862000')

#2020
get_weatherHistory(apiKey, '37.71512771387348', '-121.90834387103284', '1577862000', '1609484400')

# 10/01/2020 00:00 PT 1601535600
# 1602313200
# 08/01/2021 00:00 PT 1627801200

# 1/1/2013 1357027200
# 2/1/2013 1359705600
