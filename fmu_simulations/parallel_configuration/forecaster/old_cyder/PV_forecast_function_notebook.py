#!/usr/bin/env python
# coding: utf-8

# In[20]:


from __future__ import division
import numpy as np
import scipy as sc
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere
from pvlib.location import Location
import pvlib.irradiance as irrad
from sklearn.neural_network import MLPRegressor 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import math
from sklearn.metrics import mean_squared_error
from random import gauss
from datetime import datetime
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from sklearn.externals import joblib
from sklearn import linear_model
#import cvxpy as cvp
import pickle
import json
import pdb


# # Functions

# In[21]:


def rmse(x,y):
    N = np.prod(x.shape)
    return np.sqrt((np.sum(np.square(y-x)))/float(N))


# In[22]:


def multiStepSARIMAforecast(curDf, model_fit, predHor):
    """
    Obtain multiple-step ahead forecast using SARIMA model

    curDf: the set of up-to-now collected observations
    model_fit: fitted SARIMA model used for predictions
    predHor: prediction horizon (set to what used in the MPC)
    """

    predictions = np.empty([1, predHor])  # array with multiple-step ahead predictions
    curModel = SARIMAX(curDf, order=model_fit.specification.order,
                       seasonal_order=model_fit.specification.seasonal_order)
    #startTime = time.time()
    curModelfit = curModel.filter(model_fit.params)
    #print('It took {} seconds to build filter'.format(time.time()-startTime))

    #startTime = time.time()
    yhat = curModelfit.forecast(steps=predHor)
    #print('It took {} seconds to forecast'.format(time.time() - startTime))
    predictions = yhat.values

    return predictions


# In[23]:


def multiStepSARIMAforecast_withRetrain(curDf, model_fit, predHor, timeStep, retrainFlag):
    """
    Obtain multiple-step ahead forecast using SARIMA model
    Allows the possibility to retrain the model using predictions as 'new observations'

    curDf: the set of up-to-now collected observations
    model_fit: fitted SARIMA model used for predictions
    predHor: prediction horizon (set to what used in the MPC)
    timeStep: time step of forecasted time series (in minutes)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    """

    predictions = np.empty([1, predHor])  # array with multiple-step ahead predictions
    curDf_multistep = curDf  # auxiliary dataframe used for storing the previously generated predictions as "new observations"

    curModel = SARIMAX(curDf, order=model_fit.specification.order,
                       seasonal_order=model_fit.specification.seasonal_order)
    curModelfit = curModel.filter(model_fit.params)

    for i in range(predHor):
        # 1-step ahead prediction
        yhat = curModelfit.forecast(steps=1)
        yhat_df = pd.DataFrame(data=np.array(yhat), index=[curDf_multistep.index[-1] + pd.Timedelta(minutes=timeStep)])
        curDf_multistep = curDf_multistep.append(yhat_df)  # update auxiliary (local) prediction dataframe
        predictions[0, i] = float(yhat.values)

        # Record 1-step ahead prediction separately
        if i == 0:
            yhat_1step = pd.DataFrame(data=np.array(yhat), index=[curDf.index[-1] + pd.Timedelta(minutes=timeStep)])

        # Update SARIMA model with observation proxies (previous model predictions) up to current time step within the prediction horizon
        curModel = SARIMAX(curDf_multistep, order=model_fit.specification.order,
                           seasonal_order=model_fit.specification.seasonal_order)
        if (retrainFlag):
            curModelfit = curModel.fit()
        else:
            curModelfit = curModel.filter(model_fit.params)

    return predictions, yhat_1step


# In[24]:


def getPVforecast_v1(obsDf,SARIMAorder,SARIMAparams,predHor,timeStep,timeStepCtrl,tsPeriod,retrainFlag,NNmodel,normalizeData,wfDf,alpha,resample2CtrlStep):   
    '''
    This function is called at each MPC execution to return the forecast of the uncertain disturbance
    
    Inputs:
    obsDf: up-to-date dataframe series of total PV power (possibly normalized with rated PV power). Dataframe must have a frequency equal to timeStep.
    SARIMAorder: order of SARIMA model as (p,d,q,P,D,Q)
    SARIMAparams: parameters of SARIMA model
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    timeStepCtrl: MPC control time step (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    NNmodel: object with the fitted Neural Network model
    normalizeData: if True, I/O data are normalized; otherwise, they are not
    wfDf: dataframe with necessary weather forecasts. Dataframe has only 1 row (latest forecast)
    alpha: weighting factors to combine SARIMA and NN predictions. Final prediction=alpha*SARIMA+(1-alpha)*NN
    resample2CtrlStep: if True, the forecast will be resampled to the controller time step
    
    Outputs:
    predFinal: forecasted uncertain disturbance
    '''
    # Get SARIMA predictions
    maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
    maxLag = np.amax([1, maxSeasLag])
    SARIMAmodel = SARIMAX(obsDf[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                       seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
    SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)    
    if retrainFlag:
        predSARIMA = multiStepSARIMAforecast_withRetrain(obsDf.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor, timeStep, retrainFlag)
    else:
        predSARIMA = multiStepSARIMAforecast(obsDf.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
    predSARIMA = np.maximum(np.zeros(len(predSARIMA)),predSARIMA)
    
    # Get NN predictions
    numUpSampleNN = int(60/timeStep)
    NNinput = np.concatenate(wfDf[['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']])
    if normalizeData:
        NNinput = normalize(NNinput.reshape(1,-1))
    else:
        NNinput = NNinput.reshape(1,-1)
    tmpPred1h = np.array(NNmodel.predict(NNinput))
    
    xp1h = np.arange(1,tmpPred1h.shape[1]+1)
    xp = np.arange(1,tmpPred1h.shape[1]+1,1/float(numUpSampleNN))
    tmpPred = np.interp(xp,xp1h,tmpPred1h[0])  
    predNN = tmpPred.tolist() + tmpPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
    predNN = np.maximum(np.zeros(len(predNN)),predNN)
    
    # Combine SARIMA and NN
    predSARIMA.shape = (len(predSARIMA),1)
    predNN.shape = (len(predNN),1)
    alpha.shape = (len(alpha),1)
    pred = np.multiply(alpha,predSARIMA) + np.multiply(1-alpha,predNN)
    
    # Resample from prediction time step to MPC time step
    if resample2CtrlStep:
        if timeStep>=timeStepCtrl:
            if np.mod(timeStep,timeStepCtrl)==0:    
                numUpSample = int(timeStep/timeStepCtrl)
                predFinal = np.repeat(np.array(pred),numUpSample)
            else:
                raise ValueError('If prediction time step is larger than control time step,                 prediction time step must be an integer multiple of control time step!')
        else:
            if np.mod(timeStepCtrl,timeStep)==0:    
                numDownSample = int(timeStepCtrl/timeStep)
                predFinal = np.mean(pred.reshape(-1,numDownSample),axis=1)
            else:
                raise ValueError('If control time step is larger than prediction time step,                 control time step must be an integer multiple of prediction time step!')
    else:
        predFinal = pred
    
    predFinal.shape = (len(predFinal),1)
    return predFinal, pred, predSARIMA, predNN
    


# In[25]:


def getPVforecast(obsDf,SARIMAorder,SARIMAparams,predHor,timeStep,timeStepCtrl,tsPeriod,retrainFlag,NNmodel,normalizeData,wfDf,alpha,resample2CtrlStep):   
    '''
    This function is called at each MPC execution to return the forecast of the uncertain disturbance
    
    ***Temporary version with normalization for SARIMA and NN.
    
    Inputs:
    obsDf: up-to-date dataframe series of total PV power (possibly normalized with rated PV power). Dataframe must have a frequency equal to timeStep.
    SARIMAorder: order of SARIMA model as (p,d,q,P,D,Q)
    SARIMAparams: parameters of SARIMA model
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    timeStepCtrl: MPC control time step (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    NNmodel: object with the fitted Neural Network model
    normalizeData: if True, I/O data are normalized; otherwise, they are not
    wfDf: dataframe with necessary weather forecasts. Dataframe has only 1 row (latest forecast)
    alpha: weighting factors to combine SARIMA and NN predictions. Final prediction=alpha*SARIMA+(1-alpha)*NN
    resample2CtrlStep: if True, the forecast will be resampled to the controller time step
    
    Outputs:
    predFinal: forecasted uncertain disturbance
    '''
    
    normalizeFactor = float(2626.1504040404043)
    obsDf = obsDf/normalizeFactor
    
    # Get SARIMA predictions
    maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
    maxLag = np.amax([1, maxSeasLag])
    SARIMAmodel = SARIMAX(obsDf[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                       seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
    SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)    
    if retrainFlag:
        predSARIMA = multiStepSARIMAforecast_withRetrain(obsDf.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor, timeStep, retrainFlag)
    else:
        predSARIMA = multiStepSARIMAforecast(obsDf.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
    predSARIMA = np.maximum(np.zeros(len(predSARIMA)),predSARIMA)
    predSARIMA = predSARIMA*normalizeFactor
    
    # Get NN predictions
    numUpSampleNN = int(60/timeStep)
    NNinput = np.concatenate(wfDf[['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']])
    if normalizeData:
        NNinput = normalize(NNinput.reshape(1,-1))
    else:
        NNinput = NNinput.reshape(1,-1)
    tmpPred1h = np.array(NNmodel.predict(NNinput))
    
    xp1h = np.arange(1,tmpPred1h.shape[1]+1)
    xp = np.arange(1,tmpPred1h.shape[1]+1,1/float(numUpSampleNN))
    tmpPred = np.interp(xp,xp1h,tmpPred1h[0])  
    predNN = tmpPred.tolist() + tmpPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
    predNN = np.maximum(np.zeros(len(predNN)),predNN)
    predNN = predNN*normalizeFactor
    
    # Combine SARIMA and NN
    predSARIMA.shape = (len(predSARIMA),1)
    predNN.shape = (len(predNN),1)
    alpha.shape = (len(alpha),1)
    pred = np.multiply(alpha,predSARIMA) + np.multiply(1-alpha,predNN)
    
    # Resample from prediction time step to MPC time step
    if resample2CtrlStep:
        if timeStep>=timeStepCtrl:
            if np.mod(timeStep,timeStepCtrl)==0:    
                numUpSample = int(timeStep/timeStepCtrl)
                predFinal = np.repeat(np.array(pred),numUpSample)
            else:
                raise ValueError('If prediction time step is larger than control time step,                 prediction time step must be an integer multiple of control time step!')
        else:
            if np.mod(timeStepCtrl,timeStep)==0:    
                numDownSample = int(timeStepCtrl/timeStep)
                predFinal = np.mean(pred.reshape(-1,numDownSample),axis=1)
            else:
                raise ValueError('If control time step is larger than prediction time step,                 control time step must be an integer multiple of prediction time step!')
    else:
        predFinal = pred
    
    predFinal.shape = (len(predFinal),1)
    return predFinal, pred, predSARIMA, predNN
    


# In[26]:


def getPVforecast_v2_betaVersion(obsDf,wfDf,models,predHor=96,timeStep=15,timeStepCtrl=5,tsPeriod=96,retrainFlag=False,resample2CtrlStep=False):   
    '''    
    This function is called at each MPC execution to return the forecast of the uncertain disturbance
    
    Inputs:
    obsDf: up-to-date dataframe series of total PV power (possibly normalized with rated PV power). Dataframe must have a frequency equal to timeStep.
    models: a dictionary with the active models for forecasting
        SARIMAorder: order of SARIMA model as (p,d,q,P,D,Q)
        SARIMAparams: parameters of SARIMA model
        NNmodel: object with the fitted Neural Network model
        alpha: weighting factors to combine SARIMA and NN predictions. Final prediction=alpha*SARIMA+(1-alpha)*NN
        normPowerCoeff: coefficient to normalize PV power data
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    timeStepCtrl: MPC control time step (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    wfDf: dataframe with necessary weather forecasts. Dataframe has only 1 row (latest forecast)
    resample2CtrlStep: if True, the forecast will be resampled to the controller time step
    
    Outputs:
    predFinal: forecasted uncertain variable (sampled in controller time step)
    pred: forecasted uncertain variable (sampled in the time step of forecasting module) 
    predSARIMA: SARIMA forecast (sampled in the time step of forecasting module)
    predNN: NN forecast (sampled in the time step of forecasting module)
    '''
    
    # Get predictions from linear regression
    if models['regression']['loaded']==True:
        reg = linear_model.LinearRegression()
        histStepsRegr = models['regression']['history']
        predStepsRegr = models['regression']['prediction']
        reg.fit(np.arange(0,histStepsRegr).reshape(-1, 1),np.array(obsDf[-histStepsRegr:]))
        predRegr = reg.predict(np.arange(histStepsRegr,histStepsRegr+predStepsRegr).reshape(-1, 1))
        predRegr = np.maximum(np.zeros(len(predRegr)),predRegr)
        predRegr.shape = (len(predRegr),1)
    else:
        predRegr = np.nan
    
    # Get SARIMA predictions
    if models['sarima']['loaded']==True:
        SARIMAorder = models['sarima']['model']['SARIMAorder']
        SARIMAparams = models['sarima']['model']['SARIMAparams']
        normalizeFactor = float(models['sarima']['normPowerCoeff'])
        obsDf = obsDf/normalizeFactor
        
        maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
        maxLag = np.amax([1, maxSeasLag])
        SARIMAmodel = SARIMAX(obsDf[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                           seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
        SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)    
        if retrainFlag:
            predSARIMA = multiStepSARIMAforecast_withRetrain(obsDf.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor, timeStep, retrainFlag)
        else:
            predSARIMA = multiStepSARIMAforecast(obsDf.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
        predSARIMA = np.maximum(np.zeros(len(predSARIMA)),predSARIMA)
        predSARIMA = predSARIMA*normalizeFactor
        predSARIMA.shape = (len(predSARIMA),1)
    else:
        predSARIMA = np.nan
    
    # Get NN predictions
    if models['nn']['loaded']==True:
        NNmodel = models['nn']['model']
        normalizeData = models['nn']['normInputData']
        normalizeFactor = models['nn']['normPowerCoeff']
        numUpSampleNN = int(60/timeStep)
        NNinput = np.concatenate(wfDf[['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']])
        if normalizeData:
            NNinput = normalize(NNinput.reshape(1,-1))
        else:
            NNinput = NNinput.reshape(1,-1)
        tmpPred1h = np.array(NNmodel.predict(NNinput))

        xp1h = np.arange(1,tmpPred1h.shape[1]+1)
        xp = np.arange(1,tmpPred1h.shape[1]+1,1/float(numUpSampleNN))
        tmpPred = np.interp(xp,xp1h,tmpPred1h[0])  
        predNN = tmpPred.tolist() + tmpPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
        predNN = np.maximum(np.zeros(len(predNN)),predNN)
        predNN = predNN*normalizeFactor
        predNN.shape = (len(predNN),1)
    else:
        predNN = np.nan
    
    # Combine SARIMA and NN
    if (models['alpha']['loaded']==True) & (models['sarima']['loaded']==True) & (models['nn']['loaded']==True):
        alpha = models['alpha']['model']
        alpha.shape = (len(alpha),1)
        pred = np.multiply(alpha,predSARIMA) + np.multiply(1-alpha,predNN)
    elif (models['sarima']['loaded']==True):
        pred = predSARIMA
    elif (models['nn']['loaded']==True):
        pred = predNN
    else:
        raise ValueError('At least one forecast model must be loaded!')
    
    # Combine (SARIMA+NN) with regression model
    if models['regression']['loaded']==True:
        pred[0:predStepsRegr] = predRegr
    
    # Resample from prediction time step to MPC time step
    if resample2CtrlStep:
        if timeStep>=timeStepCtrl:
            if np.mod(timeStep,timeStepCtrl)==0:    
                numUpSample = int(timeStep/timeStepCtrl)
                predFinal = np.repeat(np.array(pred),numUpSample)
            else:
                raise ValueError('If prediction time step is larger than control time step,                 prediction time step must be an integer multiple of control time step!')
        else:
            if np.mod(timeStepCtrl,timeStep)==0:    
                numDownSample = int(timeStepCtrl/timeStep)
                predFinal = np.mean(pred.reshape(-1,numDownSample),axis=1)
            else:
                raise ValueError('If control time step is larger than prediction time step,                 control time step must be an integer multiple of prediction time step!')
    else:
        predFinal = pred
    
    predFinal.shape = (len(predFinal),1)
    return predFinal, pred, predSARIMA, predNN
    


# In[ ]:


def normalizeInputNN(models,Xf,cols_df,predHor):
    normFactTa = float(models['nn']['normTa'])
    normFactCC = float(models['nn']['normCC'])
    normFactCS = float(models['nn']['normCS'])
    normFactPdminus1 = 1 # dataframe already contains normalized power data
    
    if len(Xf.shape)==1: # reshaping needed if normalization is used when getting new forecast
        Xf.shape=(1,Xf.shape[0]) 
    
    if models['nn']['architecture'] == 'scalar':
        if cols_df == ['cloud_cover_forecast','clear_sky_forecast']:
            Xf[:,0] = Xf[:,0]/normFactCC
            Xf[:,1] = Xf[:,1]/normFactCS
        elif cols_df == ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']:
            Xf[:,0] = Xf[:,0]/normFactTa
            Xf[:,1] = Xf[:,1]/normFactCC
            Xf[:,2] = Xf[:,2]/normFactCS
        elif cols_df == ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']:
            Xf[:,0] = Xf[:,0]/normFactTa
            Xf[:,1] = Xf[:,1]/normFactCC
            Xf[:,2] = Xf[:,2]/normFactCS
            Xf[:,3] = Xf[:,3]/normFactPdminus1       
        if models['nn']['inputData'][-1]==True:
            Xf[:,-1] = Xf[:,-1]/float(predHor)
    else:
        if cols_df == ['cloud_cover_forecast','clear_sky_forecast']:
            Xf[:,0:predHor] = Xf[:,0:predHor]/normFactCC
            Xf[:,predHor:2*predHor] = Xf[:,predHor:2*predHor]/normFactCS
        elif cols_df == ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']:
            Xf[:,0:predHor] = Xf[:,0:predHor]/normFactTa
            Xf[:,predHor:2*predHor] = Xf[:,predHor:2*predHor]/normFactCC
            Xf[:,2*predHor:3*predHor] = Xf[:,2*predHor:3*predHor]/normFactCS
        elif cols_df == ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']:
            Xf[:,0:predHor] = Xf[:,0:predHor]/normFactTa
            Xf[:,predHor:2*predHor] = Xf[:,predHor:2*predHor]/normFactCC
            Xf[:,2*predHor:3*predHor] = Xf[:,2*predHor:3*predHor]/normFactCS
            Xf[:,3*predHor:4*predHor] = Xf[:,3*predHor:4*predHor]/normFactPdminus1       
        if models['nn']['inputData'][-1]==True:
            Xf[:,-predHor:] = Xf[:,-predHor:]/float(predHor)
    
    return Xf


# In[ ]:


def getPVforecast_v2(obsDf,wfDf,models,predHor=96,timeStep=15,timeStepCtrl=5,tsPeriod=96,retrainFlag=False,resample2CtrlStep=False):   
    '''    
    This function is called at each MPC execution to return the forecast of the uncertain disturbance
    
    Inputs:
    obsDf: up-to-date dataframe series of total PV power (possibly normalized with rated PV power). Dataframe must have a frequency equal to timeStep.
    models: a dictionary with the active models for forecasting
        SARIMAorder: order of SARIMA model as (p,d,q,P,D,Q)
        SARIMAparams: parameters of SARIMA model
        NNmodel: object with the fitted Neural Network model
        alpha: weighting factors to combine SARIMA and NN predictions. Final prediction=alpha*SARIMA+(1-alpha)*NN
        normPowerCoeff: coefficient to normalize PV power data
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    timeStepCtrl: MPC control time step (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    wfDf: dataframe with necessary weather forecasts. Dataframe has only 1 row (latest forecast)
    resample2CtrlStep: if True, the forecast will be resampled to the controller time step
    
    Outputs:
    predFinal: forecasted uncertain variable (sampled in controller time step)
    pred: forecasted uncertain variable (sampled in the time step of forecasting module) 
    predSARIMA: SARIMA forecast (sampled in the time step of forecasting module)
    predNN: NN forecast (sampled in the time step of forecasting module)
    '''
    
    # Get predictions from linear regression
    if models['regression']['loaded']==True:
        reg = linear_model.LinearRegression()
        histStepsRegr = models['regression']['history']
        predStepsRegr = models['regression']['prediction']
        reg.fit(np.arange(0,histStepsRegr).reshape(-1, 1),np.array(obsDf[-histStepsRegr:]))
        predRegr = reg.predict(np.arange(histStepsRegr,histStepsRegr+predStepsRegr).reshape(-1, 1))
        predRegr = np.maximum(np.zeros(len(predRegr)),predRegr)
        predRegr.shape = (len(predRegr),1)
    else:
        predRegr = np.nan
    
    # Get SARIMA predictions
    if models['sarima']['loaded']==True:
        obsDfSARIMA = obsDf.copy()
        SARIMAorder = models['sarima']['model']['SARIMAorder']
        SARIMAparams = models['sarima']['model']['SARIMAparams']
        normalizeFactor = float(models['sarima']['normPowerCoeff'])
        obsDfSARIMA = obsDfSARIMA/normalizeFactor
        maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
        maxLag = np.amax([1, maxSeasLag])
        SARIMAmodel = SARIMAX(obsDfSARIMA[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                           seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
        SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)    
        if retrainFlag:
            predSARIMA = multiStepSARIMAforecast_withRetrain(obsDfSARIMA.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor, timeStep, retrainFlag)
        else:
            predSARIMA = multiStepSARIMAforecast(obsDfSARIMA.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
        predSARIMA = np.maximum(np.zeros(len(predSARIMA)),predSARIMA)
        predSARIMA = predSARIMA*normalizeFactor
        predSARIMA.shape = (len(predSARIMA),1)
    else:
        predSARIMA = np.nan
    
    # Get NN predictions
    if models['nn']['loaded']==True:
        obsDfNN = obsDf.copy()
        NNmodel = models['nn']['model']
        normalizeData = models['nn']['normInputData']
        normalizeFactor = models['nn']['normPowerCoeff']
        numUpSampleNN = int(60/timeStep)
        predHorNN = 23
        obsDfNN = obsDfNN/normalizeFactor
        
        # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
        if models['nn']['inputData']==[False,True,True,False,False]:
            cols_df = ['cloud_cover_forecast','clear_sky_forecast']
            comb_df = wfDf.copy()
            comb_df = comb_df[cols_df]
            NNinput = np.concatenate(comb_df)
        elif (models['nn']['inputData']==[True,True,True,False,False]) | (models['nn']['inputData']==[True,True,True,False,True]):
            cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
            comb_df = wfDf.copy()
            comb_df = comb_df[cols_df]
            NNinput = np.concatenate(comb_df)
        elif (models['nn']['inputData']==[True,True,True,True,False]) | (models['nn']['inputData']==[True,True,True,True,True]):
            cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
            obsDfdminus1 = obsDfNN.loc[obsDfNN.index>obsDfNN.index[-1]-pd.Timedelta('1 days 00:00:00')]
            obsDfdminus1 = obsDfdminus1.resample('60T').mean()
            obsDfdminus1 = obsDfdminus1.tolist()
            obsDfdminus1 = obsDfdminus1[0:predHorNN]
            comb_df = wfDf.copy()
            comb_df['Observations_dminus1'] = obsDfdminus1
            NNinput = np.concatenate(comb_df[cols_df])
            
        if models['nn']['architecture'] == 'scalar':
            ncolsX = len(cols_df)
            if models['nn']['inputData'][-1]==True: ncolsX+=1
            Xf = np.empty([predHorNN,ncolsX])
            Yf = np.empty([predHorNN,1])
            idx = 0
            for j in range(predHorNN):
                tmp = []
                for k in cols_df:
                    tmp = tmp + [np.array(comb_df[k][j])]
                if models['nn']['inputData'][-1]==True:
                    tmp = tmp + [np.array(j+1)]
                Xf[idx,:] = tmp
                if normalizeData: 
                    Xf[idx,:] = normalize(Xf[idx,:].reshape(1,-1))
                Yf[idx,:] = NNmodel.predict(Xf[idx,:].reshape(1,-1)) # NN forecasts
                idx += 1
            tmpPred1h = np.array(Yf)
            tmpPred1h.shape=(1,len(tmpPred1h))
        elif models['nn']['architecture'] == 'vector':
            if models['nn']['inputData'][-1]==True:
                NNinput = np.concatenate((NNinput ,np.arange(1,predHorNN+1)))
            if normalizeData:
                NNinput = normalize(NNinput.reshape(1,-1))
            else:
                NNinput = NNinput.reshape(1,-1)
            tmpPred1h = np.array(NNmodel.predict(NNinput))
        else:
            raise ValueError('No appropriate selection for NN architecture!                     Use either scalar or vector')

        xp1h = np.arange(1,tmpPred1h.shape[1]+1)
        xp = np.arange(0.5,tmpPred1h.shape[1]+0.5,1/float(numUpSampleNN))
        tmpPred = np.interp(xp,xp1h,tmpPred1h[0])
        
        # extend the last 1h prediction, as the NN produces predictions for 23 hours only (use regression)
        #predNN = tmpPred.tolist() + tmpPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
        regNN = linear_model.LinearRegression()
        histStepsRegrNN = numUpSampleNN
        predStepsRegrNN = numUpSampleNN
        regNN.fit(np.arange(0,histStepsRegrNN).reshape(-1, 1),np.array(tmpPred[-numUpSampleNN:]))
        predRegrNN = regNN.predict(np.arange(histStepsRegrNN,histStepsRegrNN+predStepsRegrNN).reshape(-1, 1))
        predRegrNN = np.maximum(np.zeros(len(predRegrNN)),predRegrNN)
        predRegrNN.shape = (1,len(predRegrNN))
        predNN = tmpPred.tolist() + predRegrNN.tolist()[0]
        predNN = np.maximum(np.zeros(len(predNN)),predNN)
        predNN = predNN*normalizeFactor
        predNN.shape = (len(predNN),1)
    else:
        predNN = np.nan
    
    # Combine SARIMA and NN
    if (models['alpha']['loaded']==True) & (models['sarima']['loaded']==True) & (models['nn']['loaded']==True):
        alpha = models['alpha']['model']
        alpha.shape = (len(alpha),1)
        pred = np.multiply(alpha,predSARIMA) + np.multiply(1-alpha,predNN)
    elif (models['sarima']['loaded']==True):
        pred = predSARIMA
    elif (models['nn']['loaded']==True):
        pred = predNN
    else:
        raise ValueError('At least one forecast model must be loaded!')
    
    # Combine (SARIMA+NN) with regression model
    if models['regression']['loaded']==True:
        pred[0:predStepsRegr] = predRegr
    
    # Add logic to convert night values to zero
    csrad = np.array(comb_df['clear_sky_forecast'])
    csrad_upsampled = np.repeat(csrad,numUpSampleNN)
    csrad_upsampled = np.append(csrad_upsampled,csrad_upsampled[-numUpSampleNN:])
    indices = np.where(csrad_upsampled<1)
    pred[indices] = 0
    
    # Resample from prediction time step to MPC time step
    if resample2CtrlStep:
        if timeStep>=timeStepCtrl:
            if np.mod(timeStep,timeStepCtrl)==0:    
                numUpSample = int(timeStep/timeStepCtrl)
                predFinal = np.repeat(np.array(pred),numUpSample)
            else:
                raise ValueError('If prediction time step is larger than control time step,                 prediction time step must be an integer multiple of control time step!')
        else:
            if np.mod(timeStepCtrl,timeStep)==0:    
                numDownSample = int(timeStepCtrl/timeStep)
                predFinal = np.mean(pred.reshape(-1,numDownSample),axis=1)
            else:
                raise ValueError('If control time step is larger than prediction time step,                 control time step must be an integer multiple of prediction time step!')
    else:
        predFinal = pred
    
    predFinal.shape = (len(predFinal),1)
    return predFinal, pred, predSARIMA, predNN
    


# In[ ]:


def getPVforecast_v3(obsDf,wfDf,models,predHor=96,timeStep=15,timeStepCtrl=5,tsPeriod=96,retrainFlag=False,resample2CtrlStep=False):   
    '''    
    This function is called at each MPC execution to return the forecast of the uncertain disturbance
    
    Update: customized normalization for NN
    
    Inputs:
    obsDf: up-to-date dataframe series of total PV power (possibly normalized with rated PV power). Dataframe must have a frequency equal to timeStep.
    models: a dictionary with the active models for forecasting
        SARIMAorder: order of SARIMA model as (p,d,q,P,D,Q)
        SARIMAparams: parameters of SARIMA model
        NNmodel: object with the fitted Neural Network model
        alpha: weighting factors to combine SARIMA and NN predictions. Final prediction=alpha*SARIMA+(1-alpha)*NN
        normPowerCoeff: coefficient to normalize PV power data
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    timeStepCtrl: MPC control time step (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    wfDf: dataframe with necessary weather forecasts. Dataframe has only 1 row (latest forecast)
    resample2CtrlStep: if True, the forecast will be resampled to the controller time step
    
    Outputs:
    predFinal: forecasted uncertain variable (sampled in controller time step)
    pred: forecasted uncertain variable (sampled in the time step of forecasting module) 
    predSARIMA: SARIMA forecast (sampled in the time step of forecasting module)
    predNN: NN forecast (sampled in the time step of forecasting module)
    '''
    
    # Get predictions from linear regression
    if models['regression']['loaded']==True:
        reg = linear_model.LinearRegression()
        histStepsRegr = models['regression']['history']
        predStepsRegr = models['regression']['prediction']
        reg.fit(np.arange(0,histStepsRegr).reshape(-1, 1),np.array(obsDf[-histStepsRegr:]))
        predRegr = reg.predict(np.arange(histStepsRegr,histStepsRegr+predStepsRegr).reshape(-1, 1))
        predRegr = np.maximum(np.zeros(len(predRegr)),predRegr)
        predRegr.shape = (len(predRegr),1)
    else:
        predRegr = np.nan
    
    # Get SARIMA predictions
    if models['sarima']['loaded']==True:
        obsDfSARIMA = obsDf.copy()
        SARIMAorder = models['sarima']['model']['SARIMAorder']
        SARIMAparams = models['sarima']['model']['SARIMAparams']
        normalizeFactor = float(models['sarima']['normPowerCoeff'])
        obsDfSARIMA = obsDfSARIMA/normalizeFactor
        maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
        maxLag = np.amax([1, maxSeasLag])
        
        SARIMAmodel = SARIMAX(obsDfSARIMA[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                           seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
        SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)    
        
        if retrainFlag:
            predSARIMA = multiStepSARIMAforecast_withRetrain(obsDfSARIMA.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor, timeStep, retrainFlag)
        else:
            #predSARIMA = multiStepSARIMAforecast(obsDfSARIMA.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
            yhat = SARIMAmodelFit.forecast(steps=predHor)
            predSARIMA = yhat.values
        
        predSARIMA = np.maximum(np.zeros(len(predSARIMA)),predSARIMA)
        predSARIMA = predSARIMA*normalizeFactor
        predSARIMA.shape = (len(predSARIMA),1)
    else:
        predSARIMA = np.nan
    
    # Get NN predictions
    if (models['nn']['loaded']==True):
        obsDfNN = obsDf.copy()
        NNmodel = models['nn']['model']
        normalizeData = models['nn']['normInputData']
        normalizeFactOutput = models['nn']['normPowerCoeff']
        numUpSampleNN = int(60/timeStep)
        predHorNN = 23
        obsDfNN = obsDfNN/normalizeFactOutput
        
        # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
        if models['nn']['inputData']==[False,True,True,False,False]:
            cols_df = ['cloud_cover_forecast','clear_sky_forecast']
            comb_df = wfDf.copy()
            comb_df = comb_df[cols_df]
            NNinput = np.concatenate(comb_df)
        elif (models['nn']['inputData']==[True,True,True,False,False]) | (models['nn']['inputData']==[True,True,True,False,True]):
            cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
            comb_df = wfDf.copy()
            comb_df = comb_df[cols_df]
            NNinput = np.concatenate(comb_df)
        elif (models['nn']['inputData']==[True,True,True,True,False]) | (models['nn']['inputData']==[True,True,True,True,True]):
            cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
            obsDfdminus1 = obsDfNN.loc[obsDfNN.index>obsDfNN.index[-1]-pd.Timedelta('1 days 00:00:00')]
            obsDfdminus1 = obsDfdminus1.resample('60T').mean()
            obsDfdminus1 = obsDfdminus1.tolist()
            obsDfdminus1 = obsDfdminus1[0:predHorNN]
            comb_df = wfDf.copy()
            comb_df['Observations_dminus1'] = obsDfdminus1
            NNinput = np.concatenate(comb_df[cols_df])
            
        if models['nn']['architecture'] == 'scalar':
            ncolsX = len(cols_df)
            if models['nn']['inputData'][-1]==True: ncolsX+=1
            Xf = np.empty([predHorNN,ncolsX])
            Yf = np.empty([predHorNN,1])
            idx = 0
            for j in range(predHorNN):
                tmp = []
                for k in cols_df:
                    tmp = tmp + [np.array(comb_df[k][j])]
                if models['nn']['inputData'][-1]==True:
                    tmp = tmp + [np.array(j+1)]
                Xf[idx,:] = tmp
                if normalizeData: 
                    Xf[idx,:] = normalizeInputNN(models,Xf[idx,:],cols_df,predHorNN)
                Yf[idx,:] = NNmodel.predict(Xf[idx,:].reshape(1,-1)) # NN forecasts
                idx += 1
            tmpPred1h = np.array(Yf)
            tmpPred1h.shape=(1,len(tmpPred1h))
        elif models['nn']['architecture'] == 'vector':
            if models['nn']['inputData'][-1]==True:
                NNinput = np.concatenate((NNinput ,np.arange(1,predHorNN+1)))
            if normalizeData:
                NNinput = normalizeInputNN(models,NNinput,cols_df,predHorNN)
            else:
                NNinput = NNinput.reshape(1,-1)
            tmpPred1h = np.array(NNmodel.predict(NNinput))
        else:
            raise ValueError('No appropriate selection for NN architecture!                     Use either scalar or vector')

        xp1h = np.arange(1,tmpPred1h.shape[1]+1)
        xp = np.arange(0.5,tmpPred1h.shape[1]+0.5,1/float(numUpSampleNN))
        tmpPred = np.interp(xp,xp1h,tmpPred1h[0])
        
        # extend the last 1h prediction, as the NN produces predictions for 23 hours only (use regression)
        #predNN = tmpPred.tolist() + tmpPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
        regNN = linear_model.LinearRegression()
        histStepsRegrNN = numUpSampleNN
        predStepsRegrNN = numUpSampleNN
        regNN.fit(np.arange(0,histStepsRegrNN).reshape(-1, 1),np.array(tmpPred[-numUpSampleNN:]))
        predRegrNN = regNN.predict(np.arange(histStepsRegrNN,histStepsRegrNN+predStepsRegrNN).reshape(-1, 1))
        predRegrNN = np.maximum(np.zeros(len(predRegrNN)),predRegrNN)
        predRegrNN.shape = (1,len(predRegrNN))
        predNN = tmpPred.tolist() + predRegrNN.tolist()[0]
        predNN = np.maximum(np.zeros(len(predNN)),predNN)
        predNN = predNN*normalizeFactOutput
        predNN.shape = (len(predNN),1)
    else:
        predNN = np.nan
    
    # Combine SARIMA and NN
    if (models['alpha']['loaded']==True) & (models['sarima']['loaded']==True) & (models['nn']['loaded']==True):
        alpha = models['alpha']['model']
        alpha.shape = (len(alpha),1)
        pred = np.multiply(alpha,predSARIMA) + np.multiply(1-alpha,predNN)
    elif (models['sarima']['loaded']==True):
        pred = predSARIMA
    elif (models['nn']['loaded']==True):
        pred = predNN
    else:
        raise ValueError('At least one forecast model must be loaded!')
    
    # Combine (SARIMA+NN) with regression model
    if models['regression']['loaded']==True:
        pred[0:predStepsRegr] = predRegr
    
    # Add logic to convert night values to zero
    #csrad = np.array(wfDf['clear_sky_forecast'])
    #csrad_upsampled = np.repeat(csrad,numUpSampleNN)
    #csrad_upsampled = np.append(csrad_upsampled,csrad_upsampled[-numUpSampleNN:])
    #indices = np.where(csrad_upsampled<1)   
    obsDftmp = obsDf.copy()
    obsDfdminus1tmp = obsDftmp.loc[obsDftmp.index>obsDftmp.index[-1]-pd.Timedelta('1 days 00:00:00')]
    indices = np.where(np.array(obsDfdminus1tmp)<10)
    pred[indices] = 0
    
    # Resample from prediction time step to MPC time step
    if resample2CtrlStep:
        if timeStep>=timeStepCtrl:
            if np.mod(timeStep,timeStepCtrl)==0:    
                numUpSample = int(timeStep/timeStepCtrl)
                predFinal = np.repeat(np.array(pred),numUpSample)
            else:
                raise ValueError('If prediction time step is larger than control time step,                 prediction time step must be an integer multiple of control time step!')
        else:
            if np.mod(timeStepCtrl,timeStep)==0:    
                numDownSample = int(timeStepCtrl/timeStep)
                predFinal = np.mean(pred.reshape(-1,numDownSample),axis=1)
            else:
                raise ValueError('If control time step is larger than prediction time step,                 control time step must be an integer multiple of prediction time step!')
    else:
        predFinal = pred
    
    predFinal.shape = (len(predFinal),1)
    return predFinal, pred, predSARIMA, predNN
    


# In[ ]:


def getPVforecast_v4(obsDf,wfDf,models,predHor=96,timeStep=15,timeStepCtrl=5,tsPeriod=96,retrainFlag=False,resample2CtrlStep=False):   
    '''    
    This function is called at each MPC execution to return the forecast of the uncertain disturbance
    
    Update: customized normalization for NN
    
    Inputs:
    obsDf: up-to-date dataframe series of total PV power (possibly normalized with rated PV power). Dataframe must have a frequency equal to timeStep.
    models: a dictionary with the active models for forecasting
        SARIMAorder: order of SARIMA model as (p,d,q,P,D,Q)
        SARIMAparams: parameters of SARIMA model
        NNmodel: object with the fitted Neural Network model
        alpha: weighting factors to combine SARIMA and NN predictions. Final prediction=alpha*SARIMA+(1-alpha)*NN
        normPowerCoeff: coefficient to normalize PV power data
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    timeStepCtrl: MPC control time step (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    wfDf: dataframe with necessary weather forecasts. Dataframe has only 1 row (latest forecast)
    resample2CtrlStep: if True, the forecast will be resampled to the controller time step
    
    Outputs:
    predFinal: forecasted uncertain variable (sampled in controller time step)
    pred: forecasted uncertain variable (sampled in the time step of forecasting module) 
    predSARIMA: SARIMA forecast (sampled in the time step of forecasting module)
    predNN: NN forecast (sampled in the time step of forecasting module)
    '''
    
    # Get predictions from linear regression
    if models['regression']['loaded']==True:
        reg = linear_model.LinearRegression()
        histStepsRegr = models['regression']['history']
        predStepsRegr = models['regression']['prediction']
        reg.fit(np.arange(0,histStepsRegr).reshape(-1, 1),np.array(obsDf[-histStepsRegr:]))
        predRegr = reg.predict(np.arange(histStepsRegr,histStepsRegr+predStepsRegr).reshape(-1, 1))
        predRegr = np.maximum(np.zeros(len(predRegr)),predRegr)
        predRegr.shape = (len(predRegr),1)
    else:
        predRegr = np.nan
    
    # Get SARIMA predictions
    if models['sarima']['loaded']==True:
        obsDfSARIMA = obsDf.copy()
        SARIMAorder = models['sarima']['model']['SARIMAorder']
        SARIMAparams = models['sarima']['model']['SARIMAparams']
        normalizeFactor = float(models['sarima']['normPowerCoeff'])
        obsDfSARIMA = obsDfSARIMA/normalizeFactor
        maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
        maxLag = np.amax([1, maxSeasLag])
        SARIMAmodel = SARIMAX(obsDfSARIMA[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                           seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
        SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)    
        if retrainFlag:
            predSARIMA = multiStepSARIMAforecast_withRetrain(obsDfSARIMA.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor, timeStep, retrainFlag)
        else:
            predSARIMA = multiStepSARIMAforecast(obsDfSARIMA.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
        predSARIMA = np.maximum(np.zeros(len(predSARIMA)),predSARIMA)
        predSARIMA = predSARIMA*normalizeFactor
        predSARIMA.shape = (len(predSARIMA),1)
    else:
        predSARIMA = np.nan
    
    # Get NN predictions
    numUpSampleNN = int(60/timeStep)
    if ((models['nn']['loaded']==True) & (wfDf['valid'].all()==True)):
        obsDfNN = obsDf.copy()
        NNmodel = models['nn']['model']
        normalizeData = models['nn']['normInputData']
        normalizeFactOutput = models['nn']['normPowerCoeff']
        predHorNN = 23
        obsDfNN = obsDfNN/normalizeFactOutput
        
        # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
        obsDfdminus1 = obsDfNN.loc[obsDfNN.index>obsDfNN.index[-1]-pd.Timedelta('1 days 00:00:00')]
        minute_df = int(obsDfdminus1.index[-1].minute)
        if models['nn']['inputData']==[False,True,True,False,False]:
            cols_df = ['cloud_cover_forecast','clear_sky_forecast']
            comb_df = wfDf.copy()
            comb_df = comb_df[cols_df]
            NNinput = np.concatenate(comb_df)
        elif (models['nn']['inputData']==[True,True,True,False,False]) | (models['nn']['inputData']==[True,True,True,False,True]):
            cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
            comb_df = wfDf.copy()
            comb_df = comb_df[cols_df]
            NNinput = np.concatenate(comb_df)
        elif (models['nn']['inputData']==[True,True,True,True,False]) | (models['nn']['inputData']==[True,True,True,True,True]):
            cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
            obsDfdminus1 = obsDfdminus1.resample('60T',base=minute_df).mean()
            obsDfdminus1 = obsDfdminus1.tolist()
            obsDfdminus1 = obsDfdminus1[0:predHorNN]
            comb_df = wfDf.copy()
            comb_df['Observations_dminus1'] = obsDfdminus1
            NNinput = np.concatenate(comb_df[cols_df])
            
        if models['nn']['architecture'] == 'scalar':
            ncolsX = len(cols_df)
            if models['nn']['inputData'][-1]==True: ncolsX+=1
            Xf = np.empty([predHorNN,ncolsX])
            Yf = np.empty([predHorNN,1])
            idx = 0
            for j in range(predHorNN):
                tmp = []
                for k in cols_df:
                    tmp = tmp + [np.array(comb_df[k][j])]
                if models['nn']['inputData'][-1]==True:
                    tmp = tmp + [np.array(j+1)]
                Xf[idx,:] = tmp
                if normalizeData: 
                    Xf[idx,:] = normalizeInputNN(models,Xf[idx,:],cols_df,predHorNN)
                Yf[idx,:] = NNmodel.predict(Xf[idx,:].reshape(1,-1)) # NN forecasts
                idx += 1
            tmpPred1h = np.array(Yf)
            tmpPred1h.shape=(1,len(tmpPred1h))
        elif models['nn']['architecture'] == 'vector':
            if models['nn']['inputData'][-1]==True:
                NNinput = np.concatenate((NNinput ,np.arange(1,predHorNN+1)))
            if normalizeData:
                NNinput = normalizeInputNN(models,NNinput,cols_df,predHorNN)
            else:
                NNinput = NNinput.reshape(1,-1)
            tmpPred1h = np.array(NNmodel.predict(NNinput))
        else:
            raise ValueError('No appropriate selection for NN architecture!                     Use either scalar or vector')
        
        xp1h = np.arange(1,tmpPred1h.shape[1]+1)
        xp = np.arange(1,tmpPred1h.shape[1]+1,1/float(numUpSampleNN))
        tmpPred = np.interp(xp,xp1h,tmpPred1h[0])
        
        # extend the last 1h prediction, as the NN produces predictions for 23 hours only (use regression)
        #predNN = tmpPred.tolist() + tmpPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
        regNN = linear_model.LinearRegression()
        histStepsRegrNN = numUpSampleNN
        predStepsRegrNN = numUpSampleNN
        regNN.fit(np.arange(0,histStepsRegrNN).reshape(-1, 1),np.array(tmpPred[-numUpSampleNN:]))
        predRegrNN = regNN.predict(np.arange(histStepsRegrNN,histStepsRegrNN+predStepsRegrNN).reshape(-1, 1))
        predRegrNN = np.maximum(np.zeros(len(predRegrNN)),predRegrNN)
        predRegrNN.shape = (1,len(predRegrNN))
        predNN = tmpPred.tolist() + predRegrNN.tolist()[0]
        predNN = np.maximum(np.zeros(len(predNN)),predNN)
        predNN = predNN*normalizeFactOutput
        predNN.shape = (len(predNN),1)
    else:
        predNN = np.array([-1]*(predHor-4))
    
    # Combine SARIMA and NN
    if (models['alpha']['loaded']==True) & (models['sarima']['loaded']==True) & ((models['nn']['loaded']==True) & (wfDf['valid'].all()==True)):
        alpha = models['alpha']['model']
        alpha.shape = (len(alpha),1)
        pred = np.multiply(alpha,predSARIMA) + np.multiply(1-alpha,predNN)
    elif (models['sarima']['loaded']==True):
        pred = predSARIMA
    elif ((models['nn']['loaded']==True) & (wfDf['valid'].all()==True)):
        pred = predNN
    else:
        raise ValueError('At least one forecast model must be loaded!')
    
    # Combine (SARIMA+NN) with regression model
    if models['regression']['loaded']==True:
        pred[0:predStepsRegr] = predRegr
    
    # Add logic to convert night values to zero
    csrad = np.array(wfDf['clear_sky_forecast'])
    csrad_upsampled = np.repeat(csrad,numUpSampleNN)
    csrad_upsampled = np.append(csrad_upsampled,csrad_upsampled[-numUpSampleNN:])
    indices = np.where(csrad_upsampled<1)
    pred[indices] = 0
    
    # Remove the 24th hour prediction
    pred = pred[0:predHor-4]
    if models['sarima']['loaded']==True: predSARIMA = predSARIMA[0:predHor-4]
    if ((models['nn']['loaded']==True) & (wfDf['valid'].all()==True)): predNN = predNN[0:predHor-4]
    
    # Resample from prediction time step to MPC time step
    if resample2CtrlStep:
        if timeStep>=timeStepCtrl:
            if np.mod(timeStep,timeStepCtrl)==0:    
                numUpSample = int(timeStep/timeStepCtrl)
                predFinal = np.repeat(np.array(pred),numUpSample)
            else:
                raise ValueError('If prediction time step is larger than control time step,                 prediction time step must be an integer multiple of control time step!')
        else:
            if np.mod(timeStepCtrl,timeStep)==0:    
                numDownSample = int(timeStepCtrl/timeStep)
                predFinal = np.mean(pred.reshape(-1,numDownSample),axis=1)
            else:
                raise ValueError('If control time step is larger than prediction time step,                 control time step must be an integer multiple of prediction time step!')
    else:
        predFinal = pred
    
    predFinal.shape = (len(predFinal),1)
    return predFinal, pred, predSARIMA, predNN
    


# In[27]:


def retrainSARIMAmodel(obsDf,filename1,filename2, tsPeriod=96):   
    '''
    This function retrains the params of SARIMA model.
    The model structure (p,d,q,P,D,Q orders) is fixed to the previously trained model.
    
    Inputs:   
    obsDf: up-to-date dataframe series of total PV power (possibly normalized with rated PV power). Dataframe must have a frequency equal to timeStep.
    filename1: file with current trained model
    filename2: file to write the newly trained model
    
    Outputs:
    SARIMA_model_updated (SARIMAorder, SARIMAparams): newly trained model
    '''

    with open(filename1, 'rb') as f:
        SARIMAres = json.load(f)
    SARIMAorder = np.array(SARIMAres['order']) # fix the already identified optimal order structure
    SARIMA_params_old = np.array(SARIMAres['params'])

    train = obsDf.copy()
    model = SARIMAX(train, order=(SARIMAorder[0], SARIMAorder[1], SARIMAorder[2]), 
                    seasonal_order=(SARIMAorder[3], SARIMAorder[4], SARIMAorder[5], tsPeriod))
    model_fit = model.fit(disp=0, method='lbfgs',start_params=SARIMA_params_old)
    SARIMAparams = model_fit.params

    SARIMA_model_updated = {'order':SARIMAorder.tolist(), 'params':SARIMAparams.get_values().tolist()}
    with open(filename2, 'wb') as f:
        json.dump(SARIMA_model_updated, f)
        
    return SARIMAorder, SARIMAparams


# In[28]:


def makeCombinedDf_NN_v1(wf_df,obsDf):
    '''
    This function puts together the weather data and historical power data 
    in the format that the NN training function expects.
    
    Inputs:
    wf_df: historical weather forecast dataframe
    obsDf: historical power dataframe (dataframe has only one data column)
    wf_df and obsDf have both the same size and frequency of 1 hour
    
    Outputs:
    comb_df: the combined dataframe
    '''
    predHor = int(len(wf_df['Tamb_forecast'].iloc[0])) # NN prediction horizon
    
    i = 0
    dfs_overlap = True
    while (obsDf.index[i]!=wf_df.index[0]): 
        i+=1
        if i>len(obsDf):
            dfs_overlap = False
    if (dfs_overlap==False) | (len(obsDf[i:])<predHor):
        raise ValueError('The weather forecast and power measurements dataframes do not overlap!')
    
    numPoints = np.amin([len(wf_df),len(obsDf.iloc[i:-predHor])]) # number of input-output pairs for NN training
    comb_df = wf_df.iloc[0:numPoints]
    print(len(wf_df))
    print(len(comb_df))
    Parray = []
    cnt = 0
    for idx in range(i,len(obsDf)-predHor):
        Parray = Parray + [obsDf.iloc[idx:idx+predHor].tolist()]
        cnt+=1
        if cnt>=len(wf_df): break
    comb_df['Observations'] = Parray # power measurements
    
    return comb_df


# In[29]:


def makeCombinedDf_NN(wf_df,obsDf):
    '''
    This function puts together the weather data and historical power data 
    in the format that the NN training function expects.
    
    Inputs:
    wf_df: historical weather forecast dataframe
    obsDf: historical power dataframe (dataframe has only one data column)
    wf_df and obsDf have both the same size and frequency of 1 hour
    
    Outputs:
    comb_df: the combined dataframe
    '''
    predHor = int(len(wf_df[wf_df.columns[0]].iloc[0])) # NN prediction horizon
    intersect = np.intersect1d(obsDf.index, wf_df.index)
    comb_df = wf_df.loc[intersect]
    #tmpP = obsDf
    tmpP = obsDf.loc[intersect]
    Parray = []
    Parray_dminus1 = []
    dropIdx = []
    for idx in range(len(comb_df.index)):
        #if (sum([comb_df.index[idx]-pd.Timedelta('0 days '+str(hourIdx)+':00:00') in tmpP.index for hourIdx in range(1,predHor+1)])==predHor):
        if (sum([comb_df.index[idx]+pd.Timedelta('0 days '+str(hourIdx)+':00:00') in tmpP.index for hourIdx in range(1,predHor+1)])==predHor) & (sum([comb_df.index[idx]-pd.Timedelta('0 days '+str(hourIdx)+':00:00') in tmpP.index for hourIdx in range(1,predHor+1)])==predHor):
            Parray = Parray + [tmpP.loc[((tmpP.index>=comb_df.index[idx]+pd.Timedelta('0 days 01:00:00')) &
                                              (tmpP.index<=comb_df.index[idx]+pd.Timedelta('0 days '+str(hourIdx)+':00:00')))].tolist()]
            Parray_dminus1 = Parray_dminus1 + [tmpP.loc[((tmpP.index<=comb_df.index[idx]-pd.Timedelta('0 days 01:00:00')) &
                                              (tmpP.index>=comb_df.index[idx]-pd.Timedelta('0 days '+str(hourIdx)+':00:00')))].tolist()]
        else:
            dropIdx.append(idx)
    comb_df = comb_df.drop(comb_df.index[dropIdx])
    comb_df['Observations'] = Parray
    comb_df['Observations_dminus1'] = Parray_dminus1

    return comb_df


# In[30]:


def retrainNNmodel(models, comb_df, hyperParams, normInputData, normOutputFact, filename, filenameAux, testSetPerc=0.8):
    '''
    This function retrains the params of the NN model. 
    
    Inputs:
    models: prediction models
    comb_df: dataframe that combines weather data and historical observations (power data)
    hyperParams: dictionary with the grid search for hyper parameters (the range of each hyper-parameter is given as a list).
    normInputData: If True, the NN input data are normalized before training
    normOutputFact: the factor with which the historical observations are scaled before training
    testSetPerc: proportion (from 0 to 1) of comb_df that is used for validation/testing
    filename: the filename to save the newly trained NN model
    '''
    
    # Split observation dataframe to training and validation sets
    predHor = int(len(comb_df[comb_df.columns[0]].iloc[0])) # NN prediction horizon
    trainSetPerc = 1-testSetPerc
    trainSize = int(trainSetPerc*len(comb_df))
    trainSet = []
    if (np.isnan(models['nn']['randomSeed'])==False): 
        np.random.seed(seed=models['nn']['randomSeed'])
    while (len(trainSet)<trainSize):
        newElem = int(np.random.uniform(0,len(comb_df),1))
        if newElem not in trainSet:
            trainSet = trainSet + [newElem]
    trainSet = np.sort(trainSet)
    trainNN_df = comb_df.iloc[trainSet]
    testSet = np.setdiff1d(range(len(comb_df)),trainSet)
    testNN_df = comb_df.iloc[testSet]
    
    # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
    if models['nn']['inputData']==[False,True,True,False,False]:
        cols_df = ['cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,False,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    elif models['nn']['inputData']==[True,True,True,False,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    
    # Select NN architecture
    if models['nn']['architecture']=='scalar':
        nrows = len(trainNN_df)*predHor
        ncolsX = int(len(np.concatenate(trainNN_df[cols_df].iloc[0]))/float(predHor))
        if models['nn']['inputData'][-1]==True: ncolsX+=1
        ncolsY = 1
        X = np.empty([nrows,ncolsX])
        Y = np.empty([nrows,ncolsY])
        idx = 0
        for i in range(len(trainNN_df)):
            for j in range(predHor):
                tmp = []
                for k in cols_df:
                    tmp = tmp + [np.array(trainNN_df[k].iloc[i][j])]
                if models['nn']['inputData'][-1]==True: tmp = tmp + [np.array(j+1)]
                X[idx,:] = tmp
                Y[idx,:] = np.array(trainNN_df['Observations'].iloc[i][j])
                idx += 1
        Y = Y.ravel() 
        if normInputData:
            X = normalize(X)      
    elif models['nn']['architecture']=='vector':
        # Prepare and normalize input/output data 
        nrows = len(trainNN_df)
        ncolsX = len(np.concatenate(trainNN_df[cols_df].iloc[0]))
        if models['nn']['inputData'][-1]==True: ncolsX+=predHor
        ncolsY = predHor
        X = np.empty([nrows,ncolsX])
        for i in range(nrows):
            tmp = np.concatenate(trainNN_df[cols_df].iloc[i])
            if models['nn']['inputData'][-1]==True:
                tmp = np.concatenate((tmp ,np.arange(1,predHor+1)))
            X[i,:] = tmp
        Y = np.empty([nrows,ncolsY])
        for i in range(nrows):
            Y[i,:] = trainNN_df['Observations'].iloc[i]
        if normInputData:
            X = normalize(X)
    else:
        raise ValueError('No appropriate selection for NN architecture!                 Use either scalar or vector')
    idx=0
    numCases_lbfgs = len(hyperParams['hidden_layer_sizes'])*len(hyperParams['activation'])*len(hyperParams['max_iter'])*len(hyperParams['early_stopping'])*len(hyperParams['validation_fraction'])
    numCases_adam = len(hyperParams['hidden_layer_sizes'])*len(hyperParams['activation'])*len(hyperParams['max_iter'])*len(hyperParams['early_stopping'])*len(hyperParams['validation_fraction'])
    numCases_sgd = len(hyperParams['hidden_layer_sizes'])*len(hyperParams['activation'])*len(hyperParams['max_iter'])*len(hyperParams['early_stopping'])*len(hyperParams['validation_fraction'])*len(hyperParams['learning_rate'])
    if 'lbfgs' in hyperParams['solver']: lbfgs=1
    else: lbfgs=0
    if 'sgd' in hyperParams['solver']: sgd=1
    else: sgd=0
    if 'adam' in hyperParams['solver']: adam=1
    else: adam=0    
    numCases_tot = lbfgs*numCases_lbfgs + adam*numCases_adam + sgd*numCases_sgd
    NNmodel_list = []
    RMSE_insample_list = []
    for hidden_layer_sizes in hyperParams['hidden_layer_sizes']:
        for activation in hyperParams['activation']:
            for solver in hyperParams['solver']:
                for learning_rate in hyperParams['learning_rate']:
                    for max_iter in hyperParams['max_iter']:
                        for early_stopping in hyperParams['early_stopping']:
                            for validation_fraction in hyperParams['validation_fraction']:
                                if ((solver=='sgd') | (((solver=='lbfgs') | (solver=='adam')) & (learning_rate=='constant'))):
                                    NNmodel = MLPRegressor(solver=solver,hidden_layer_sizes=hidden_layer_sizes,activation=activation,max_iter=max_iter,
                                                           validation_fraction=validation_fraction,learning_rate=learning_rate,early_stopping=early_stopping)
                                    NNmodel.fit(X,Y)
                                    NNmodel_list  = NNmodel_list + [NNmodel]
                                    Yf_insample = NNmodel.predict(X)
                                    RMSE_insample_list = RMSE_insample_list + [rmse(Y,Yf_insample)]
                                    idx+=1
                                    print('Percentage (%) finished for given architecture: {}'.format(100*np.round(idx/float(numCases_tot),3)))
    RMSE_list = []
    RMSE_day_list = [] # RMSE only for daytime
    
    for NNmodel in NNmodel_list:
        # Forecast with NN (out-of-sample)
        if models['nn']['architecture']=='scalar':
            nrows = len(testNN_df)*predHor
            ncolsX = int(len(np.concatenate(testNN_df[cols_df].iloc[0]))/float(predHor))
            if models['nn']['inputData'][-1]==True: ncolsX+=1
            ncolsY = 1
            Xf = np.empty([nrows,ncolsX])
            Yf = np.empty([nrows,ncolsY])
            Yr = np.empty([nrows,ncolsY])
            idx = 0
            for i in range(len(testNN_df)):
                for j in range(predHor):
                    tmp = []
                    for k in cols_df:
                        tmp = tmp + [np.array(testNN_df[k].iloc[i][j])]
                    if models['nn']['inputData'][-1]==True:
                        tmp = tmp + [np.array(j+1)]
                    Xf[idx,:] = tmp
                    if normInputData: 
                        Xf[idx,:] = normalize(Xf[idx,:].reshape(1,-1))
                    Yf[idx,:] = NNmodel.predict(Xf[idx,:].reshape(1,-1)) # NN forecasts
                    Yf[idx,:] = normOutputFact*Yf[idx,:] # de-normalize
                    Yr[idx,:] = normOutputFact*testNN_df['Observations'].iloc[i][j]
                    idx += 1
        elif models['nn']['architecture']=='vector':
            nrows = len(testNN_df)
            ncolsX = len(np.concatenate(testNN_df[cols_df].iloc[0]))
            if models['nn']['inputData'][-1]==True: ncolsX+=predHor
            ncolsY = len(testNN_df['Observations'].iloc[0])
            Xf = np.empty([nrows,ncolsX])
            Yf = np.empty([nrows,ncolsY])
            Yr = np.empty([nrows,ncolsY])
            for i in range(nrows):
                tmp = np.concatenate(testNN_df[cols_df].iloc[i])
                if models['nn']['inputData'][-1]==True:
                    tmp = np.concatenate((tmp ,np.arange(1,predHor+1)))
                Xf[i,:] = tmp
                if normInputData:
                    Xf[i,:] = normalize(Xf[i,:].reshape(1,-1))
                Yf[i,:] = NNmodel.predict(Xf[i,:].reshape(1,-1)) # NN forecasts
                Yf[i,:] = normOutputFact*Yf[i,:] # de-normalize
                Yr[i,:] = np.array(normOutputFact)*testNN_df['Observations'].iloc[i]
        else:
            raise ValueError('No appropriate selection for NN architecture!                     Use either scalar or vector')
        RMSE_list = RMSE_list + [rmse(Yr,Yf)]
        daytime_idx = np.where(Yr>10) # larger than 10 Watt
        RMSE_day_list = RMSE_day_list + [rmse(Yr[daytime_idx],Yf[daytime_idx])]
    
    RMSE_list_sorted_index = np.argsort(RMSE_list)
    bestRMSE_index = RMSE_list_sorted_index[0]
    NNmodel_best = NNmodel_list[bestRMSE_index]
    joblib.dump(NNmodel_best, filename)
    
    with open('RMSE_insample_list_'+filenameAux+'.json', 'wb') as f:
        json.dump(RMSE_insample_list, f)
    with open('RMSE_list_'+filenameAux+'.json', 'wb') as f:
        json.dump(RMSE_list, f)
    with open('RMSE_day_list_'+filenameAux+'.json', 'wb') as f:
        json.dump(RMSE_day_list, f)
    joblib.dump(NNmodel_list, 'NNmodel_list_'+filenameAux+'.sav')
    
    return NNmodel_best, RMSE_insample_list, RMSE_list, RMSE_day_list, NNmodel_list


# In[ ]:


def retrainNNmodel_v2(models, comb_df, hyperParams, normInputData, normOutputFact, filename, filenameAux, testSetPerc=0.2):
    '''
    This function retrains the params of the NN model. 
    
    Update: customized normalization for NN
    
    Inputs:
    models: prediction models
    comb_df: dataframe that combines weather data and historical observations (power data)
    hyperParams: dictionary with the grid search for hyper parameters (the range of each hyper-parameter is given as a list).
    normInputData: If True, the NN input data are normalized before training
    normOutputFact: the factor with which the historical observations are scaled before training
    testSetPerc: proportion (from 0 to 1) of comb_df that is used for validation/testing
    filename: the filename to save the newly trained NN model
    '''
    
    # Split observation dataframe to training and validation sets
    predHor = int(len(comb_df[comb_df.columns[0]].iloc[0])) # NN prediction horizon
    trainSetPerc = 1-testSetPerc
    trainSize = int(trainSetPerc*len(comb_df))
    trainSet = []
    if (np.isnan(models['nn']['randomSeed'])==False): 
        np.random.seed(seed=models['nn']['randomSeed'])
    while (len(trainSet)<trainSize):
        newElem = int(np.random.uniform(0,len(comb_df),1))
        if newElem not in trainSet:
            trainSet = trainSet + [newElem]
    trainSet = np.sort(trainSet)
    trainNN_df = comb_df.iloc[trainSet]
    testSet = np.setdiff1d(range(len(comb_df)),trainSet)
    testNN_df = comb_df.iloc[testSet]
    
    # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
    if models['nn']['inputData']==[False,True,True,False,False]:
        cols_df = ['cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,False,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    elif models['nn']['inputData']==[True,True,True,False,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    
    # Select NN architecture
    if models['nn']['architecture']=='scalar':
        nrows = len(trainNN_df)*predHor
        ncolsX = int(len(np.concatenate(trainNN_df[cols_df].iloc[0]))/float(predHor))
        if models['nn']['inputData'][-1]==True: ncolsX+=1
        ncolsY = 1
        X = np.empty([nrows,ncolsX])
        Y = np.empty([nrows,ncolsY])
        idx = 0
        notnanIdx = []
        for i in range(len(trainNN_df)):
            for j in range(predHor):
                tmp = []
                for k in cols_df:
                    tmp = tmp + [np.array(trainNN_df[k].iloc[i][j])]
                if models['nn']['inputData'][-1]==True: tmp = tmp + [np.array(j+1)]
                X[idx,:] = tmp
                Y[idx,:] = np.array(trainNN_df['Observations'].iloc[i][j])
                if (len(np.where(np.isnan(X[idx,:]))[0])==0) & (~np.isnan(Y[idx,:])):
                    notnanIdx = notnanIdx+[idx]
                idx += 1
        X = X[notnanIdx,:]
        Y = Y[notnanIdx,:]
        Y = Y.ravel() 
        if normInputData:
            X = normalizeInputNN(models,X,cols_df,predHor) 
    elif models['nn']['architecture']=='vector':
        # Prepare and normalize input/output data 
        nrows = len(trainNN_df)
        ncolsX = len(np.concatenate(trainNN_df[cols_df].iloc[0]))
        if models['nn']['inputData'][-1]==True: ncolsX+=predHor
        ncolsY = predHor
        X = np.empty([nrows,ncolsX])
        Y = np.empty([nrows,ncolsY])
        notnanIdx = []
        for i in range(nrows):
            tmp = np.concatenate(trainNN_df[cols_df].iloc[i])
            if models['nn']['inputData'][-1]==True:
                tmp = np.concatenate((tmp ,np.arange(1,predHor+1)))
            X[i,:] = tmp
            Y[i,:] = trainNN_df['Observations'].iloc[i]
            if (len(np.where(np.isnan(X[i,:]))[0])==0) & (len(np.where(np.isnan(Y[i,:]))[0])==0):
                notnanIdx = notnanIdx+[i]
        X = X[notnanIdx,:]
        Y = Y[notnanIdx,:]
        if normInputData:
            X = normalizeInputNN(models,X,cols_df,predHor) 
    else:
        raise ValueError('No appropriate selection for NN architecture!                 Use either scalar or vector')
    idx=0
    numCases_lbfgs = len(hyperParams['hidden_layer_sizes'])*len(hyperParams['activation'])*len(hyperParams['max_iter'])*len(hyperParams['early_stopping'])*len(hyperParams['validation_fraction'])
    numCases_adam = len(hyperParams['hidden_layer_sizes'])*len(hyperParams['activation'])*len(hyperParams['max_iter'])*len(hyperParams['early_stopping'])*len(hyperParams['validation_fraction'])
    numCases_sgd = len(hyperParams['hidden_layer_sizes'])*len(hyperParams['activation'])*len(hyperParams['max_iter'])*len(hyperParams['early_stopping'])*len(hyperParams['validation_fraction'])*len(hyperParams['learning_rate'])
    if 'lbfgs' in hyperParams['solver']: lbfgs=1
    else: lbfgs=0
    if 'sgd' in hyperParams['solver']: sgd=1
    else: sgd=0
    if 'adam' in hyperParams['solver']: adam=1
    else: adam=0    
    numCases_tot = lbfgs*numCases_lbfgs + adam*numCases_adam + sgd*numCases_sgd
    NNmodel_list = []
    RMSE_insample_list = []
    for hidden_layer_sizes in hyperParams['hidden_layer_sizes']:
        for activation in hyperParams['activation']:
            for solver in hyperParams['solver']:
                for learning_rate in hyperParams['learning_rate']:
                    for max_iter in hyperParams['max_iter']:
                        for early_stopping in hyperParams['early_stopping']:
                            for validation_fraction in hyperParams['validation_fraction']:
                                if ((solver=='sgd') | (((solver=='lbfgs') | (solver=='adam')) & (learning_rate=='constant'))):
                                    NNmodel = MLPRegressor(solver=solver,hidden_layer_sizes=hidden_layer_sizes,activation=activation,max_iter=max_iter,
                                                           validation_fraction=validation_fraction,learning_rate=learning_rate,early_stopping=early_stopping)
                                    NNmodel.fit(X,Y)
                                    NNmodel_list  = NNmodel_list + [NNmodel]
                                    Yf_insample = NNmodel.predict(X)
                                    RMSE_insample_list = RMSE_insample_list + [rmse(Y,Yf_insample)]
                                    idx+=1
                                    print('Percentage (%) finished for given architecture: {}'.format(100*np.round(idx/float(numCases_tot),3)))
    RMSE_list = []
    RMSE_day_list = [] # RMSE only for daytime
    
    for NNmodel in NNmodel_list:
        # Forecast with NN (out-of-sample)
        if models['nn']['architecture']=='scalar':
            nrows = len(testNN_df)*predHor
            ncolsX = int(len(np.concatenate(testNN_df[cols_df].iloc[0]))/float(predHor))
            if models['nn']['inputData'][-1]==True: ncolsX+=1
            ncolsY = 1
            Xf = np.empty([nrows,ncolsX])
            Yf = np.empty([nrows,ncolsY])
            Yr = np.empty([nrows,ncolsY])
            idx = 0
            notnanIdx = []
            for i in range(len(testNN_df)):
                for j in range(predHor):
                    tmp = []
                    for k in cols_df:
                        tmp = tmp + [np.array(testNN_df[k].iloc[i][j])]
                    if models['nn']['inputData'][-1]==True:
                        tmp = tmp + [np.array(j+1)]
                    Xf[idx,:] = tmp
                    if (len(np.where(np.isnan(Xf[idx,:]))[0])==0):
                        notnanIdx = notnanIdx+[idx]
                        if normInputData: 
                            Xf[idx,:] = normalizeInputNN(models,Xf[idx,:],cols_df,predHor)
                        Yf[idx,:] = NNmodel.predict(Xf[idx,:].reshape(1,-1)) # NN forecasts
                        Yf[idx,:] = normOutputFact*Yf[idx,:] # de-normalize
                        Yr[idx,:] = normOutputFact*testNN_df['Observations'].iloc[i][j]
                    idx += 1
            Xf = Xf[notnanIdx,:]
            Yf = Yf[notnanIdx,:]
            Yr = Yr[notnanIdx,:]
        elif models['nn']['architecture']=='vector':
            nrows = len(testNN_df)
            ncolsX = len(np.concatenate(testNN_df[cols_df].iloc[0]))
            if models['nn']['inputData'][-1]==True: ncolsX+=predHor
            ncolsY = len(testNN_df['Observations'].iloc[0])
            Xf = np.empty([nrows,ncolsX])
            Yf = np.empty([nrows,ncolsY])
            Yr = np.empty([nrows,ncolsY])
            notnanIdx = []
            for i in range(nrows):
                tmp = np.concatenate(testNN_df[cols_df].iloc[i])
                if models['nn']['inputData'][-1]==True:
                    tmp = np.concatenate((tmp ,np.arange(1,predHor+1)))
                Xf[i,:] = tmp
                if (len(np.where(np.isnan(Xf[i,:]))[0])==0):
                    notnanIdx = notnanIdx+[i]
                    if normInputData:
                        Xf[i,:] = normalizeInputNN(models,Xf[i,:],cols_df,predHor)
                    Yf[i,:] = NNmodel.predict(Xf[i,:].reshape(1,-1)) # NN forecasts
                    Yf[i,:] = normOutputFact*Yf[i,:] # de-normalize
                    Yr[i,:] = np.array(normOutputFact)*testNN_df['Observations'].iloc[i]
            Xf = Xf[notnanIdx,:]
            Yf = Yf[notnanIdx,:]
            Yr = Yr[notnanIdx,:]
        else:
            raise ValueError('No appropriate selection for NN architecture!                     Use either scalar or vector')
        RMSE_list = RMSE_list + [rmse(Yr,Yf)]
        daytime_idx = np.where(Yr>10) # larger than 10 Watt
        RMSE_day_list = RMSE_day_list + [rmse(Yr[daytime_idx],Yf[daytime_idx])]
    
    RMSE_list_sorted_index = np.argsort(RMSE_list)
    bestRMSE_index = RMSE_list_sorted_index[0]
    NNmodel_best = NNmodel_list[bestRMSE_index]
    joblib.dump(NNmodel_best, filename)
    
    with open('RMSE_insample_list_'+filenameAux+'.json', 'wb') as f:
        json.dump(RMSE_insample_list, f)
    with open('RMSE_list_'+filenameAux+'.json', 'wb') as f:
        json.dump(RMSE_list, f)
    with open('RMSE_day_list_'+filenameAux+'.json', 'wb') as f:
        json.dump(RMSE_day_list, f)
    joblib.dump(NNmodel_list, 'NNmodel_list_'+filenameAux+'.sav')
    
    return NNmodel_best, RMSE_insample_list, RMSE_list, RMSE_day_list, NNmodel_list


# In[31]:


def optimizeWeights_v1(SARIMAorder,SARIMAparams,initSet,trainSet,predHor,timeStep,tsPeriod,
                    NNmodel,normalizeData,normalizeFactorSARIMA,normalizeFactorNN,wf_df,filename):
    """
    Use fitted SARIMA and NN models and identify the optimal weighting factors 
    to combine the two forecasts into one forecast.

    SARIMAorder: list of orders of SARIMA model
    SARIMAparams: list of params of SARIMA model
    initSet: initialization set (used just to redefine the SARIMA structure with the "filter" method)
    trainSet: training set
    predHor: prediction horizon for the multiple-step ahead forecasts
    tsPreTrainFlageriod: seasonality period (number of time steps for the SARIMA model)
    NNmodel: trained NN model
    normalizeData: if True, I/O data are normalized; otherwise, they are not
    wf_df: weather data dataframe
    filename: filename to save the newly computed weighting factors
    """
    # Get individual predictions
    simHor = len(trainSet) - predHor
    obsMat = []
    obsDf = initSet
    predMatSARIMA = []
    predMatNN = []
    numUpSampleNN = int(60/timeStep)
    maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
    maxLag = np.amax([1, maxSeasLag])
    SARIMAmodel = SARIMAX(initSet[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                       seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
    SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)
    
    idx = 0
    for t in range(simHor):
        if (idx<len(wf_df)) & (trainSet.index[t].minute == 0) & (len(np.where(obsDf.isnull().iloc[-maxLag*predHor-1:])[0])==0) & (trainSet.isnull().iloc[t:t + predHor].any() == False):
            # Find the next entry in wf_df that corresponds to the same hour as in trainSet
            while (idx<len(wf_df)) & (wf_df.index[idx]!=trainSet.index[t]): 
                idx+=1
            if (idx>=len(wf_df)): break
            
            # Get SARIMA predictions
            curPred = multiStepSARIMAforecast(obsDf.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
            curPred = np.maximum(np.zeros(len(curPred)),curPred)
            curPred = curPred*normalizeFactorSARIMA
            predMatSARIMA = predMatSARIMA + [curPred.tolist()]
                   
            
            # Get NN predictions
            NNinput = np.concatenate(wf_df[['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']].iloc[idx])
            if normalizeData: 
                NNinput = normalize(NNinput.reshape(1,-1))
            else:
                NNinput = NNinput.reshape(1,-1)
            curPred1h = np.array(NNmodel.predict(NNinput))       
            xp1h = np.arange(1,curPred1h.shape[1]+1)
            xp = np.arange(1,curPred1h.shape[1]+1,1/float(numUpSampleNN))
            curPred = np.interp(xp,xp1h,curPred1h[0])
            idx+=1    
            curPred = curPred.tolist() + curPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
            curPred = np.maximum(np.zeros(len(curPred)),curPred)
            curPred = curPred*normalizeFactorNN
            predMatNN = predMatNN + [curPred]
            
            # Collect observations
            obsMat = obsMat + [trainSet.iloc[t:t + predHor].values]

        # Update 'observation' dataframe
        obs = pd.DataFrame(data=np.array([trainSet.iloc[t]]),index=[obsDf.index[-1] + pd.Timedelta(minutes=timeStep)])
        obsDf = obsDf.append(obs)

    predMatSARIMA = np.array(predMatSARIMA)
    predMatNN = np.array(predMatNN)
    obsMat = np.array(obsMat)
    
    predBlkSARIMA = np.diag(predMatSARIMA[0])
    predBlkNN = np.diag(predMatNN[0])
    numCols = predMatSARIMA.shape[0]

    for i in range(1,numCols):
        predBlkSARIMA = np.vstack((predBlkSARIMA,np.diag(predMatSARIMA[i])))
        predBlkNN = np.vstack((predBlkNN,np.diag(predMatNN[i])))
    
    obsVec = np.reshape(obsMat, (obsMat.shape[0] * obsMat.shape[1],1))
    
    # Combine predictions and optimize weighting factor
    alpha = cvp.Variable(predHor,1)
    
    residual = predBlkSARIMA*alpha + predBlkNN*(1-alpha) - obsVec
    obj = cvp.norm(residual,2)
    const = [alpha>=0, alpha<=1]
    prob = cvp.Problem(cvp.Minimize(obj), const)
    prob.solve(solver='ECOS')
    weightFact = alpha.value
    objValue = obj.value
    
    weightFactDict = {'alpha':weightFact.tolist()}
    with open(filename, 'wb') as f:
        json.dump(weightFactDict, f) 
    return weightFact
    


# In[ ]:


def optimizeWeights_v2(models,SARIMAorder,SARIMAparams,initSet,trainSet,predHor,timeStep,tsPeriod,
                    NNmodel,normalizeData,normalizeFactorSARIMA,normalizeFactorNN,wf_df,filename):
    """
    Use fitted SARIMA and NN models and identify the optimal weighting factors 
    to combine the two forecasts into one forecast.

    SARIMAorder: list of orders of SARIMA model
    SARIMAparams: list of params of SARIMA model
    initSet: initialization set (used just to redefine the SARIMA structure with the "filter" method)
    trainSet: training set
    predHor: prediction horizon for the multiple-step ahead forecasts
    tsPreTrainFlageriod: seasonality period (number of time steps for the SARIMA model)
    NNmodel: trained NN model
    normalizeData: if True, I/O data are normalized; otherwise, they are not
    wf_df: weather data dataframe
    filename: filename to save the newly computed weighting factors
    """
    # Get individual predictions
    simHor = len(trainSet) - predHor
    obsMat = []
    obsDf = pd.Series(initSet) # obsDf is un-normalized
    initSet = initSet/float(normalizeFactorSARIMA)
    predMatSARIMA = []
    predMatNN = []
    numUpSampleNN = int(60/timeStep)
    maxSeasLag = np.amax([SARIMAorder[2],SARIMAorder[5]])
    maxLag = np.amax([1, maxSeasLag])
    SARIMAmodel = SARIMAX(initSet[-maxLag*predHor-1:], order=(int(SARIMAorder[0]), int(SARIMAorder[1]), int(SARIMAorder[2])),
                       seasonal_order=(int(SARIMAorder[3]), int(SARIMAorder[4]), int(SARIMAorder[5]), tsPeriod))
    SARIMAmodelFit = SARIMAmodel.filter(SARIMAparams)
    
    idx = 0
    predHorNN = 23
    for t in range(simHor):
        print('Iteration {} of {}'.format(t, simHor))
        if (idx<len(wf_df)) & (trainSet.index[t].minute == 0) & (len(np.where(obsDf.isnull().iloc[-maxLag*predHor-1:])[0])==0) & (len(obsDf.loc[obsDf.index>obsDf.index[-1]-pd.Timedelta('1 days 00:00:00')])==96) & (trainSet.isnull().iloc[t:t + predHor].any() == False):
            # Find the next entry in wf_df that corresponds to the same hour as in trainSet
            while (idx<len(wf_df)) & (wf_df.index[idx]!=trainSet.index[t]): 
                idx+=1
            if (idx>=len(wf_df)): break
            
            # Get SARIMA predictions
            obsDfSARIMA = obsDf.copy()/float(normalizeFactorSARIMA)
            curPred = multiStepSARIMAforecast(obsDfSARIMA.iloc[-maxLag*predHor-1:], SARIMAmodelFit, predHor)
            curPred = np.maximum(np.zeros(len(curPred)),curPred)
            curPred = curPred*normalizeFactorSARIMA
            predMatSARIMA = predMatSARIMA + [curPred.tolist()]
                   
            # Get NN predictions
            obsDfNN = obsDf.copy()/float(normalizeFactorNN)
            numUpSampleNN = int(60/timeStep)

            # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
            if models['nn']['inputData']==[False,True,True,False,False]:
                cols_df = ['cloud_cover_forecast','clear_sky_forecast']
                NNinput = np.concatenate(wf_df[cols_df].iloc[idx])
            elif (models['nn']['inputData']==[True,True,True,False,False]) | (models['nn']['inputData']==[True,True,True,False,True]):
                cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
                NNinput = np.concatenate(wf_df[cols_df].iloc[idx])
            elif (models['nn']['inputData']==[True,True,True,True,False]) | (models['nn']['inputData']==[True,True,True,True,True]):
                cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
                obsDfdminus1 = obsDfNN.loc[obsDfNN.index>obsDfNN.index[-1]-pd.Timedelta('1 days 00:00:00')]
                minute_df = int(obsDfdminus1.index[-1].minute)
                obsDfdminus1 = obsDfdminus1.resample('60T',base=minute_df).mean()
                obsDfdminus1 = obsDfdminus1.tolist()
                obsDfdminus1 = obsDfdminus1[0:predHorNN]
                comb_df = wf_df.iloc[idx]
                comb_df['Observations_dminus1'] = obsDfdminus1
                NNinput = np.concatenate(comb_df[cols_df])

            if models['nn']['architecture'] == 'scalar':
                ncolsX = len(cols_df)
                if models['nn']['inputData'][-1]==True: ncolsX+=1
                Xf = np.empty([predHorNN,ncolsX])
                Yf = np.empty([predHorNN,1])
                idx2 = 0
                for j in range(predHorNN):
                    tmp = []
                    for k in cols_df:
                        tmp = tmp + [np.array(comb_df[k][j])]
                    if models['nn']['inputData'][-1]==True:
                        tmp = tmp + [np.array(j+1)]
                    Xf[idx2,:] = tmp
                    if normalizeData: 
                        Xf[idx2,:] = normalizeInputNN(models,Xf[idx2,:],cols_df,predHorNN)
                    Yf[idx2,:] = NNmodel.predict(Xf[idx2,:].reshape(1,-1)) # NN forecasts
                    idx2 += 1
                tmpPred1h = np.array(Yf)
                tmpPred1h.shape=(1,len(tmpPred1h))
            elif models['nn']['architecture'] == 'vector':
                if models['nn']['inputData'][-1]==True:
                    NNinput = np.concatenate((NNinput ,np.arange(1,predHorNN+1)))
                if normalizeData:
                    NNinput = normalizeInputNN(models,NNinput,cols_df,predHorNN)
                tmpPred1h = np.array(NNmodel.predict(NNinput.reshape(1,-1)))
            else:
                raise ValueError('No appropriate selection for NN architecture!                         Use either scalar or vector')

            xp1h = np.arange(1,tmpPred1h.shape[1]+1)
            xp = np.arange(1,tmpPred1h.shape[1]+1,1/float(numUpSampleNN))
            tmpPred = np.interp(xp,xp1h,tmpPred1h[0])

            # extend the last 1h prediction, as the NN produces predictions for 23 hours only (use regression)
            #predNN = tmpPred.tolist() + tmpPred[-numUpSampleNN:].tolist() # extend the last 1h prediction, as the NN produces predictions for 23 hours only
            regNN = linear_model.LinearRegression()
            histStepsRegrNN = numUpSampleNN
            predStepsRegrNN = numUpSampleNN
            regNN.fit(np.arange(0,histStepsRegrNN).reshape(-1, 1),np.array(tmpPred[-numUpSampleNN:]))
            predRegrNN = regNN.predict(np.arange(histStepsRegrNN,histStepsRegrNN+predStepsRegrNN).reshape(-1, 1))
            predRegrNN = np.maximum(np.zeros(len(predRegrNN)),predRegrNN)
            predRegrNN.shape = (1,len(predRegrNN))
            predNN = tmpPred.tolist() + predRegrNN.tolist()[0]

            predNN = np.maximum(np.zeros(len(predNN)),predNN)
            predNN = predNN*normalizeFactorNN
            predNN.shape = (len(predNN))
            predMatNN = predMatNN + [predNN]
            
            # Collect observations
            obsMat = obsMat + [trainSet.iloc[t:t + predHor].values]
            idx+=1

        # Update 'observation' dataframe
        obs = pd.Series(data=np.array([trainSet.iloc[t]]),index=[obsDf.index[-1] + pd.Timedelta(minutes=timeStep)])
        obsDf = obsDf.append(obs)

    if idx>0:
        predMatSARIMA = np.array(predMatSARIMA)
        predMatNN = np.array(predMatNN)
        obsMat = np.array(obsMat)

        predBlkSARIMA = np.diag(predMatSARIMA[0])
        predBlkNN = np.diag(predMatNN[0])
        numCols = predMatSARIMA.shape[0]

        for i in range(1,numCols):
            predBlkSARIMA = np.vstack((predBlkSARIMA,np.diag(predMatSARIMA[i])))
            predBlkNN = np.vstack((predBlkNN,np.diag(predMatNN[i])))

        obsVec = np.reshape(obsMat, (obsMat.shape[0] * obsMat.shape[1],1))

        # Combine predictions and optimize weighting factor
        alpha = cvp.Variable(predHor,1)

        residual = predBlkSARIMA*alpha + predBlkNN*(1-alpha) - obsVec
        obj = cvp.norm(residual,2)
        const = [alpha>=0, alpha<=1]
        prob = cvp.Problem(cvp.Minimize(obj), const)
        prob.solve(solver='ECOS')
        weightFact = alpha.value
        objValue = obj.value

        weightFactDict = {'alpha':weightFact.tolist()}
        with open(filename, 'wb') as f:
            json.dump(weightFactDict, f) 
        return weightFact
    


# In[32]:


def retrainCombinedModel(models,obsDf,wf_df,filenameSARIMAnew,filenameNNnew,filenameNNaux,filenameAlphaNew,testSetPercNN,
                         hyperParams={'hidden_layer_sizes': [(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),
                                                            (2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100),(110,110),(120,120)],
                                      'activation': ['identity','relu','logistic','tanh'],
                                      'solver': ['lbfgs','sgd','adam'],'learning_rate': ['constant','invscaling','adaptive'],'max_iter': [10000],
                                      'early_stopping': [True], 'validation_fraction': [0.1]},
                         predHor=96,timeStep=15,tsPeriod=96):
    '''
    Wrapper function to re-train the combined prediction model
    Step 1: re-train SARIMA
    Step 2: re-train NN
    Step 3: re-optimize weighting factors
    
    Inputs:
    models: prediction models
    obsDf: historical power dataframe (dataframe has only one data column)
    wf_df: historical weather forecast dataframe
    filenameSARIMAnew: filename to save newly trained SARIMA model
    filenameNNnew: filename to save newly trained NN model
    filenameAlphaNew: filename to save newly calculated alpha factors
    testSetPercNN: proportion (from 0 to 1) of comb_df that is used for validation/testing
    hyperParams: dictionary with the grid search for hyper parameters (the range of each hyper-parameter is given as a list).
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    '''
    
    # Retrain SARIMA
    startTime = time.time()
    filenameSARIMAold = models['sarima']['path']
    normalizeFactorSARIMA = float(models['sarima']['normPowerCoeff'])
    obsDf_SARIMA = obsDf.copy()/normalizeFactorSARIMA
    SARIMAorder, SARIMAparams = retrainSARIMAmodel(obsDf_SARIMA,filenameSARIMAold,filenameSARIMAnew,tsPeriod)
    print('SARIMA training: done in {} seconds'.format(np.round(time.time()-startTime)))
    
    # Retrain NN
    startTime = time.time()
    normalizeFactorNN = float(models['nn']['normPowerCoeff'])
    normInputData = models['nn']['normInputData']
    obsDf_NN = obsDf.copy()/normalizeFactorNN
    obsDf_NN = obsDf_NN.resample('60T',base=obsDf_NN.index[0].minute).mean()
    comb_df = makeCombinedDf_NN(wf_df,obsDf_NN)
    NNmodel, RMSE_insample_list, RMSE_list, RMSE_day_list, NNmodel_list = retrainNNmodel_v2(models,comb_df,hyperParams,normInputData,normalizeFactorNN,filenameNNnew,filenameNNaux,testSetPercNN)
    print('NN training: done in {} seconds'.format(np.round(time.time()-startTime)))
    
    startTime = time.time()
    initPerc = 0.05 # use 5% of available data for SARIMA model initialization
    initSize = int(initPerc*len(obsDf))
    initSet, trainSet = obsDf.iloc[0:initSize], obsDf.iloc[initSize:len(obsDf)]
    alpha = optimizeWeights_v2(models,SARIMAorder,SARIMAparams,initSet,trainSet,predHor,timeStep,
                    tsPeriod,NNmodel,normInputData,normalizeFactorSARIMA,normalizeFactorNN,wf_df,filenameAlphaNew)
    print('alpha training: done in {} seconds'.format(np.round(time.time()-startTime)))


# In[ ]:


def retrainAlphaModel(models,obsDf,wf_df,filenameSARIMAnew,filenameNNnew,filenameNNaux,filenameAlphaNew,testSetPercNN,
                         hyperParams={'hidden_layer_sizes': [(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),
                                                            (2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100),(110,110),(120,120)],
                                      'activation': ['identity','relu','logistic','tanh'],
                                      'solver': ['lbfgs','sgd','adam'],'learning_rate': ['constant','invscaling','adaptive'],'max_iter': [10000],
                                      'early_stopping': [True], 'validation_fraction': [0.1]},
                         predHor=96,timeStep=15,tsPeriod=96):
    '''
    Re-optimize weighting factors
    
    Inputs:
    models: prediction models
    obsDf: historical power dataframe (dataframe has only one data column)
    wf_df: historical weather forecast dataframe
    filenameSARIMAnew: filename to save newly trained SARIMA model
    filenameNNnew: filename to save newly trained NN model
    filenameAlphaNew: filename to save newly calculated alpha factors
    testSetPercNN: proportion (from 0 to 1) of comb_df that is used for validation/testing
    hyperParams: dictionary with the grid search for hyper parameters (the range of each hyper-parameter is given as a list).
    predHor: prediction horizon (set to what is used in the MPC), in number of time steps
    timeStep: time step of forecasted time series with SARIMA (in minutes)
    tsPeriod: seasonality in SARIMA (in number of time steps)
    '''
      
    SARIMAorder = models['sarima']['model']['SARIMAorder']
    SARIMAparams = models['sarima']['model']['SARIMAparams']
    normalizeFactorSARIMA = float(models['sarima']['normPowerCoeff'])
    normalizeFactorNN = float(models['nn']['normPowerCoeff'])
    normInputData = models['nn']['normInputData']    
    NNmodel = models['nn']['model']
    
    startTime = time.time()
    initPerc = 0.05 # use 5% of available data for SARIMA model initialization
    initSize = int(initPerc*len(obsDf))
    initSet, trainSet = obsDf.iloc[0:initSize], obsDf.iloc[initSize:len(obsDf)]
    alpha = optimizeWeights_v2(models,SARIMAorder,SARIMAparams,initSet,trainSet,predHor,timeStep,
                    tsPeriod,NNmodel,normInputData,normalizeFactorSARIMA,normalizeFactorNN,wf_df,filenameAlphaNew)
    print('alpha training: done in {} seconds'.format(np.round(time.time()-startTime)))


# In[33]:


def retrainNN(models,obsDf,wf_df,filenameNNnew,filenameNNaux,testSetPercNN,
                         hyperParams={'hidden_layer_sizes': [(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),
                                                            (2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100),(110,110),(120,120)],
                                      'activation': ['identity','relu','logistic','tanh'],
                                      'solver': ['lbfgs','sgd','adam'],'learning_rate': ['constant','invscaling','adaptive'],'max_iter': [10000],
                                      'early_stopping': [True], 'validation_fraction': [0.1]}):
    '''   
    Inputs:
    models: prediction models
    obsDf: historical power dataframe (dataframe has only one data column)
    wf_df: historical weather forecast dataframe
    filenameNNnew: filename to save newly trained NN model
    testSetPercNN: proportion (from 0 to 1) of comb_df that is used for validation/testing
    hyperParams: dictionary with the grid search for hyper parameters (the range of each hyper-parameter is given as a list).
    '''
        
    # Retrain NN
    startTime = time.time()
    normalizeFactorNN = float(models['nn']['normPowerCoeff'])
    normInputData = models['nn']['normInputData']
    obsDf_NN = obsDf/normalizeFactorNN
    aux = obsDf_NN.copy()
    aux.index = pd.to_datetime(obsDf_NN.index)
    obsDf_NN = aux.resample('60T',base=obsDf_NN.index[0].minute).mean()
    obsDf_NN = obsDf_NN.dropna()
    comb_df = makeCombinedDf_NN(wf_df,obsDf_NN)
    NNmodel, RMSE_insample_list, RMSE_list, RMSE_day_list, NNmodel_list = retrainNNmodel_v2(models,comb_df,hyperParams,normInputData,normalizeFactorNN,filenameNNnew,filenameNNaux,testSetPercNN)
    print('NN training: done in {} seconds'.format(np.round(time.time()-startTime)))


# In[ ]:


def evaluate_NN_outofsample(models,testNN_df):
    NNmodel = models['nn']['model']
    predHor = int(len(testNN_df[testNN_df.columns[0]].iloc[0])) # NN prediction horizon
    normOutputFact = float(models['nn']['normPowerCoeff'])
    normInputData = models['nn']['normInputData']
    
    # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
    if models['nn']['inputData']==[False,True,True,False,False]:
        cols_df = ['cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,False,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    elif models['nn']['inputData']==[True,True,True,False,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    
    if models['nn']['architecture']=='scalar':
        nrows = len(testNN_df)*predHor
        ncolsX = int(len(np.concatenate(testNN_df[cols_df].iloc[0]))/float(predHor))
        if models['nn']['inputData'][-1]==True: ncolsX+=1
        ncolsY = 1
        Xf = np.empty([nrows,ncolsX])
        Yf = np.empty([nrows,ncolsY])
        Yr = np.empty([nrows,ncolsY])
        idx = 0
        for i in range(len(testNN_df)):
            for j in range(predHor):
                tmp = []
                for k in cols_df:
                    tmp = tmp + [np.array(testNN_df[k].iloc[i][j])]
                if models['nn']['inputData'][-1]==True: tmp = tmp + [np.array(j+1)]
                Xf[idx,:] = tmp
                if normInputData: Xf[idx,:] = normalize(Xf[idx,:].reshape(1,-1))
                Yf[idx,:] = NNmodel.predict(Xf[idx,:].reshape(1,-1)) # NN forecasts
                Yf[idx,:] = normOutputFact*Yf[idx,:] # de-normalize
                Yr[idx,:] = normOutputFact*testNN_df['Observations'].iloc[i][j]
                idx += 1
    elif models['nn']['architecture']=='vector':
        nrows = len(testNN_df)
        ncolsX = len(np.concatenate(testNN_df[cols_df].iloc[0]))
        if models['nn']['inputData'][-1]==True: ncolsX+=predHor
        ncolsY = len(testNN_df['Observations'].iloc[0])
        Xf = np.empty([nrows,ncolsX])
        Yf = np.empty([nrows,ncolsY])
        Yr = np.empty([nrows,ncolsY])
        for i in range(nrows):
            tmp = np.concatenate(testNN_df[cols_df].iloc[i])
            if models['nn']['inputData'][-1]==True:
                tmp = np.concatenate((tmp ,np.arange(1,predHor+1)))
            Xf[i,:] = tmp
            if normInputData: Xf[i,:] = normalize(Xf[i,:].reshape(1,-1))
            Yf[i,:] = NNmodel.predict(Xf[i,:].reshape(1,-1)) # NN forecasts
            Yf[i,:] = normOutputFact*Yf[i,:] # de-normalize
            Yr[i,:] = np.array(normOutputFact)*testNN_df['Observations'].iloc[i]
    else:
        raise ValueError('No appropriate selection for NN architecture!                 Use either scalar or vector')
    RMSE = rmse(Yr,Yf)
    daytime_idx = np.where(Yr>10) # larger than 10 Watt
    RMSE_day = rmse(Yr[daytime_idx],Yf[daytime_idx])
    
    return RMSE, RMSE_day, Yr, Yf


# In[ ]:


def evaluate_NN_outofsample_v2(models,testNN_df):
    NNmodel = models['nn']['model']
    predHor = int(len(testNN_df[testNN_df.columns[0]].iloc[0])) # NN prediction horizon
    normOutputFact = float(models['nn']['normPowerCoeff'])
    normInputData = models['nn']['normInputData']
    
    # Select input data. Order: ambient temp, cloud cover, clear sky, Pdminus1, predHorizon
    if models['nn']['inputData']==[False,True,True,False,False]:
        cols_df = ['cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,False,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,False]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    elif models['nn']['inputData']==[True,True,True,False,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
    elif models['nn']['inputData']==[True,True,True,True,True]:
        cols_df = ['Tamb_forecast','cloud_cover_forecast','clear_sky_forecast','Observations_dminus1']
    
    if models['nn']['architecture']=='scalar':
        nrows = len(testNN_df)*predHor
        ncolsX = int(len(np.concatenate(testNN_df[cols_df].iloc[0]))/float(predHor))
        if models['nn']['inputData'][-1]==True: ncolsX+=1
        ncolsY = 1
        Xf = np.empty([nrows,ncolsX])
        Yf = np.empty([nrows,ncolsY])
        Yr = np.empty([nrows,ncolsY])
        idx = 0
        for i in range(len(testNN_df)):
            for j in range(predHor):
                tmp = []
                for k in cols_df:
                    tmp = tmp + [np.array(testNN_df[k].iloc[i][j])]
                if models['nn']['inputData'][-1]==True: tmp = tmp + [np.array(j+1)]
                Xf[idx,:] = tmp
                if normInputData: 
                    Xf[idx,:] = normalizeInputNN(models,Xf[idx,:],cols_df,predHor)
                Yf[idx,:] = NNmodel.predict(Xf[idx,:].reshape(1,-1)) # NN forecasts
                Yf[idx,:] = normOutputFact*Yf[idx,:] # de-normalize
                Yr[idx,:] = normOutputFact*testNN_df['Observations'].iloc[i][j]
                idx += 1
    elif models['nn']['architecture']=='vector':
        nrows = len(testNN_df)
        ncolsX = len(np.concatenate(testNN_df[cols_df].iloc[0]))
        if models['nn']['inputData'][-1]==True: ncolsX+=predHor
        ncolsY = len(testNN_df['Observations'].iloc[0])
        Xf = np.empty([nrows,ncolsX])
        Yf = np.empty([nrows,ncolsY])
        Yr = np.empty([nrows,ncolsY])
        for i in range(nrows):
            tmp = np.concatenate(testNN_df[cols_df].iloc[i])
            if models['nn']['inputData'][-1]==True:
                tmp = np.concatenate((tmp ,np.arange(1,predHor+1)))
            Xf[i,:] = tmp
            if normInputData: 
                Xf[i,:] = normalizeInputNN(models,Xf[i,:],cols_df,predHor)
            Yf[i,:] = NNmodel.predict(Xf[i,:].reshape(1,-1)) # NN forecasts
            Yf[i,:] = normOutputFact*Yf[i,:] # de-normalize
            Yr[i,:] = np.array(normOutputFact)*testNN_df['Observations'].iloc[i]
    else:
        raise ValueError('No appropriate selection for NN architecture!                 Use either scalar or vector')
    RMSE = rmse(Yr,Yf)
    daytime_idx = np.where(Yr>10) # larger than 10 Watt
    RMSE_day = rmse(Yr[daytime_idx],Yf[daytime_idx])
    
    return RMSE, RMSE_day, Yr, Yf


# In[44]:


# Load PV data
def load_pv_data(pv_data_file):
    df_pv = pd.read_csv(pv_data_file,index_col=[0])

    if pv_data_file == 'PV_data_cleanedup_20180904.csv':
        # Set index
        df_pv['date'] = pd.to_datetime(df_pv['date'])
        df_pv = df_pv.set_index('date')

        # df_pv = df_pv['totPower'] # use this if total PV power is of interest
        df_pv = df_pv['PV1_activePower_computed'] # use this if only inverter 1 PV power is of interest
        
        # resample for SARIMA
        df_pv = df_pv.resample('15T',base=df_pv.index[0].minute).mean()
        df_pv = df_pv.dropna()
        tsPeriod = 96
        
        # resample for NN
        df_pv_1h = df_pv.resample('60T',base=df_pv.index[0].minute).mean()
        df_pv_1h = df_pv_1h.dropna()
    else:
        # set index
        df_pv.index = pd.to_datetime(df_pv.index)
        
        # resample for SARIMA
        df_pv = df_pv.resample('15T',base=df_pv.index[0].minute).mean()
        df_pv = df_pv.dropna()
        tsPeriod = 96
        df_pv = df_pv['DC_PV_W']
        
        # resample for NN
        df_pv_1h = df_pv.resample('60T',base=df_pv.index[0].minute).mean()
        df_pv_1h = df_pv_1h.dropna()

    return df_pv, df_pv_1h


# In[42]:


def load_wf_data(filename):
    # Load weather forecast data
    import csv
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        data = {}
        for row in reader:
            row = ''.join(row)
            if row[0:4] == '2018':
                acq_date = row
            if row[0] == '[':
                if ((acq_date[5:7] != '07') | ((acq_date[5:7] == '07') & (int(acq_date[8:10]) >= 20))):
                    data[acq_date] = eval(row.replace('][','],['))

    # Keep only data tha correspond to full hours (to avoid redundant data)
    wf_df_aux = data
    for key in wf_df_aux.keys():
        if pd.to_datetime(key).minute!=0:
            wf_df_aux.pop(key, None)  
    return wf_df_aux


# In[1]:


def wf_data_to_df(filename,lbnl,surface_tilt,surface_azimuth,wf_df_aux,reSaveData):
    # Transform weather data dict to a dataframe with [clear sky irrad, ambient temp, cloud cover] 

    if reSaveData:
        timestampStringList = []
        Tamb_list = []
        cloudCover_list = []
        clearSky_list = []
        i = 0
        start = time.time()
        for key in wf_df_aux.keys():
            for item in list(wf_df_aux[key][2:]): 
                if item[0] not in timestampStringList:
                    timestampStringList = timestampStringList + [item[0]]
        timestampList = sorted(pd.to_datetime(timestampStringList))
        #timestamp = pd.DatetimeIndex(timestampList,tz=lbnl.tz) 
        timestamp = pd.DatetimeIndex(timestampList,tz=-3600*7) # PST is 7 hours behind UTC (lbnl object is in UTC)
        cs = lbnl.get_clearsky(timestamp)
        solarPos = lbnl.get_solarposition(timestamp)
        totIrr = irrad.total_irrad(surface_tilt,surface_azimuth,solarPos.apparent_zenith,solarPos.azimuth,cs['dni'],cs['ghi'],cs['dhi'])
        ghiPVTrain = totIrr['poa_global']

        wfdTimestamp = []
        for key in wf_df_aux.keys():
            cur_Tamb_list = []
            cur_cloudCover_list = []
            cur_clearSky_list = []
            wfdTimestamp = wfdTimestamp + [key]
            print('Percent done: {}'.format(np.round(100*i/len(wf_df_aux.keys()),2)))
            for item in list(wf_df_aux[key][2:]):   
                pos = timestampList.index(pd.to_datetime(item[0]))
                cur_Tamb_list = cur_Tamb_list + [item[1]]
                cur_cloudCover_list = cur_cloudCover_list + [item[2]]
                cur_clearSky_list = cur_clearSky_list + [ghiPVTrain.iloc[pos]]
            Tamb_list = Tamb_list + [cur_Tamb_list]
            cloudCover_list = cloudCover_list + [cur_cloudCover_list]
            clearSky_list = clearSky_list + [cur_clearSky_list]
            i+=1 
        Tamb_list = np.array(Tamb_list)   
        cloudCover_list = np.array(cloudCover_list)
        clearSky_list = np.array(clearSky_list)
        #np.savez('weather_forecast_data.npz',wfdTimestamp=wfdTimestamp,Tamb=Tamb_list,cloudCover=cloudCover_list,clearSky=clearSky_list)

        # Save sorted data in a df
        wf_df = pd.DataFrame(data=[np.array(wfdTimestamp), np.array(Tamb_list), np.array(cloudCover_list), np.array(clearSky_list)]).transpose()
        wf_df.columns = ['timestamp','Tamb_forecast','cloud_cover_forecast','clear_sky_forecast']
        aux = np.array(wf_df['timestamp'])
        wf_df = wf_df.set_index(pd.to_datetime(wf_df['timestamp']))
        wf_df = wf_df.iloc[:,[1,2,3]]
        wf_df['timeIndex'] = aux
        wf_df = wf_df.sort_values(by='timeIndex')
        wf_df.to_json(filename)

        timeelapsed = time.time()-start
        print('Time elapsed: {}'.format(timeelapsed))
        
        return wf_df


# In[ ]:


def optSARIMAstruct(train,tsPeriod,listOfOrdersToTry,filename,listOfParamsWS=None):
    """
    Iterate of a list of possible SARIMA model structures and compute in-sample RMSEs for 1-step ahead prediction
    
    train: training set (dataframe column that corresponds to total PV power)
    tsPeriod: seasonality in SARIMA
    listOfOrdersToTry: list of SARIMA structures to be checked in the form [p,d,q,P,D,Q]
    listOfParamsWS: list of initial guesses of optimal SARIMA coefficients from previous runs (WS: Warm Start)
    filename: file to save the results
    """
    
    listOfRMSEs = []
    listOfOptTimes = []
    listOfOrders = []
    listOfParams = []
    if listOfParamsWS is not None: params = listOfParamsWS
    totComb = len(listOfOrdersToTry)
    cnt = 0
    successCnt = 0
    
    for i in range(len(listOfOrdersToTry)):
        curList = listOfOrdersToTry[i]
        p = int(curList[0])
        d = int(curList[1])
        q = int(curList[2])
        P = int(curList[3])
        D = int(curList[4])
        Q = int(curList[5])
        cnt += 1
        
        try:
            startTime = time.time()
            print('Fitting model: {}'.format([p,d,q,P,D,Q]))
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, tsPeriod))
            if listOfParamsWS is not None:
                model_fit = model.fit(disp=0, method='lbfgs',start_params=params[i])
            else:
                model_fit = model.fit(disp=0, method='lbfgs')
            listOfRMSEs.append(rmse(model_fit.fittedvalues,train))
            listOfOrders.append([p,d,q,P,D,Q])
            successCnt += 1
            listOfParams.append(model_fit.params.get_values().tolist())
            optTime = time.time() - startTime
            listOfOptTimes.append(optTime)
            print('Optimization time was {} seconds. Success ratio is {}%'.format(np.round(optTime,2),np.round(100*(successCnt/cnt))))
        except:
            print('Failed!')
        print('Completed: {}%'.format(np.round(100*(cnt/totComb),2)))
    
    res = {'listOfRMSEs':listOfRMSEs, 'listOfOrders':listOfOrders, 'listOfOptTimes':listOfOptTimes, 'listOfParams':listOfParams}
    with open(filename+'.json', 'wb') as f:
        json.dump(res, f)


# In[ ]:


def findBestSARIMAOutOfSampleMultipleSteps(filename1, filename2, trainSet, testSet, tsPeriod, predHor, timeStep,
                                           retrainFlag, numModels):
    """
    Load result from grid search and identify the best SARIMA models in terms of out-of-sample performance
    Multiple-step ahead forecasts are considered

    filename1: file with saved results from 1-step ahead out-of-sample analysis
    filename2: file to save results from multiple-step ahead out-of-sample analysis
    trainSet: training set (dataframe column)
    testSet: test set (dataframe column)
    tsPeriod: seasonality in SARIMA
    predHor: prediction horizon as # of timeStep (set to what used in the MPC)
    timeStep: time step of forecasted time series (in minutes)
    retrainFlag: If True, a new SARIMA model is fit every time a new prediction is needed within the prediction horizon
    numModels: the number of top-performing models to be selected
    """

    print('=============== {}-step ahead, out-of-sample Results ==============='.format(predHor))
    with open(filename1,'rb') as f:
        data = json.load(f)
    listOfOrdersSaved = np.array(data['listOfOrders'])
    listOfRMSEsSaved = np.array(data['listOfRMSEs'])
    listOfParamsSaved = data['listOfParams']

    listOfRMSEsSaved = np.array(listOfRMSEsSaved)
    ordered = np.argsort(listOfRMSEsSaved)
    listOfOrdersSavedRanked = listOfOrdersSaved[ordered[:]]
    listOfParamsSavedRanked = []
    for i in ordered: 
        listOfParamsSavedRanked = listOfParamsSavedRanked + [listOfParamsSaved[i]]
    listOfRMSEsSavedRanked = listOfRMSEsSaved[ordered[:]]
    
    listOfRMSEsComputed = []
    numModels = np.amin([numModels, len(listOfParamsSaved)])
    simHor = len(testSet) - predHor
    totComb = numModels
    cnt = 0
    
    for i in range(numModels):
        curOrder = listOfOrdersSavedRanked[i]
        curParams = np.array(listOfParamsSavedRanked[i])
        predMat = []
        obsMat = []
        obsDf = trainSet
        print(curOrder)

        maxSeasLag = np.amax([curOrder[2],curOrder[5]])
        maxLag = np.amax([1, maxSeasLag])
        curModel = SARIMAX(trainSet[-maxLag*predHor-1:], order=(int(curOrder[0]), int(curOrder[1]), int(curOrder[2])),
                           seasonal_order=(int(curOrder[3]), int(curOrder[4]), int(curOrder[5]), tsPeriod))
        startTime = time.time()
        curModelFit = curModel.filter(curParams)
        print('It took {} seconds to build filter'.format(time.time() - startTime))

        try:
            for t in range(simHor):
                #if (testSet.index[t].hour == 6) & (testSet.index[t].minute == 0) & (len(np.where(obsDf.isnull().iloc[-maxLag*predHor-1:])[0])==0) & (testSet.isnull().iloc[t:t + predHor].any() == False):  # | (testSet.index[t].hour==12):
                if (testSet.index[t].minute == 0) & (len(np.where(obsDf.isnull().iloc[-maxLag * predHor - 1:])[0]) == 0) & (testSet.isnull().iloc[t:t + predHor].any() == False):                
                    # Get predictions
                    if retrainFlag:
                        curPred = multiStepSARIMAforecast_withRetrain(obsDf.iloc[-maxLag*predHor-1:], curModelFit, predHor, timeStep, retrainFlag)
                    else:
                        curPred = multiStepSARIMAforecast(obsDf.iloc[-maxLag*predHor-1:], curModelFit, predHor)
                    predMat = predMat + [curPred.tolist()]
                    # Collect observations
                    obsMat = obsMat + [testSet.iloc[t:t + predHor].values]

                # Update 'observation' dataframe
                obs = pd.DataFrame(data=np.array([testSet.iloc[t]]),
                                   index=[obsDf.index[-1] + pd.Timedelta(minutes=timeStep)])
                obsDf = obsDf.append(obs)
            predMat = np.array(predMat)
            obsMat = np.array(obsMat)
            predVec = np.reshape(predMat, (predMat.shape[0] * predMat.shape[1], 1))
            obsVec = np.reshape(obsMat, (obsMat.shape[0] * obsMat.shape[1], 1))
            listOfRMSEsComputed.append(rmse(predVec, obsVec))
        except:
            listOfRMSEsComputed.append(np.inf)
        cnt += 1
        print('Completed: {}%'.format(np.round(100 * (cnt / totComb), 2)))

    listOfRMSEsComputed = np.array(listOfRMSEsComputed)
    ordered = np.argsort(listOfRMSEsComputed)
    bestNordersList = listOfOrdersSavedRanked[ordered[0:numModels]]
    bestNparamsList = []
    for jj in ordered[0:numModels]:
        bestNparamsList = bestNparamsList + [listOfParamsSavedRanked[jj]]
    bestNRMSEsList = listOfRMSEsComputed[ordered[0:numModels]]
    print('The best {} model orders are:'.format(numModels))
    print(listOfOrdersSavedRanked[ordered[0:numModels]])
    print('The best {} RMSEs are:'.format(numModels))
    print(listOfRMSEsComputed[ordered[0:numModels]])

    bestOrder = listOfOrdersSavedRanked[ordered[0]]
    bestParams = listOfParamsSavedRanked[ordered[0]]
    minRMSE = listOfRMSEsComputed[ordered[0]]
    bestModel = SARIMAX(trainSet, order=(int(bestOrder[0]), int(bestOrder[1]), int(bestOrder[2])),
                        seasonal_order=(int(bestOrder[3]), int(bestOrder[4]), int(bestOrder[5]), tsPeriod))
    bestModelFit = bestModel.filter(bestParams)

    print('Results for the best model')
    print(bestModelFit.summary())
    print('\nMinumum out-of-sample nRMSE for {}-step prediction is {}'.format(predHor, minRMSE))
    
    ordersListRes = bestNordersList.tolist()
    paramsListRes = []
    for elem in bestNparamsList: 
        paramsListRes = paramsListRes + [elem]
    RMSEsListRes = bestNRMSEsList.tolist()
    res = {'listOfOrders':ordersListRes, 'listOfParams':paramsListRes, 'listOfRMSEs':RMSEsListRes}
    with open(filename2+'.json', 'wb') as f:
        json.dump(res, f)

    return bestModel, bestModelFit, bestOrder, bestParams, bestNordersList, bestNparamsList

