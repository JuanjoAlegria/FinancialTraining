# -*- coding: utf-8 -*-
import os
import numpy as np

basePath = os.path.dirname(os.path.realpath(__file__))

def loadFinancialData():
    xTrainDir = "../Datos/mondayTuesday2014_s1.npy"
    yTrainDir = "../Datos/y2014_avg.npy"
    xValidationDir = "../Datos/x2015_both.npy"
    yValidationDir = "../Datos/y2015_avg.npy"

    xTrain = np.load(os.path.join(basePath, xTrainDir))
    yTrain = np.load(os.path.join(basePath, yTrainDir))

    xValidation = np.load(os.path.join(basePath, xValidationDir))
    yValidation = np.load(os.path.join(basePath, yValidationDir))



    trainData = [xTrain, yTrain]
    validationData = [xValidation[:35], yValidation[:35]]
    testData = [[], []]

    return trainData, validationData, testData


def loadFinancialData_mondayTuesday(yDay):
    def inner_mondayTuesday():
        xTrainDir = "../Datos/mondayTuesday2014_s1.npy"
        yTrainDir = "../Datos/" + yDay + "2014_s1.npy"
        xValidationDir = "../Datos/mondayTuesday2015_s1.npy"
        yValidationDir = "../Datos/" + yDay + "2015_s1.npy"

        xTrain = np.load(os.path.join(basePath, xTrainDir))
        yTrain = np.load(os.path.join(basePath, yTrainDir))

        xValidation = np.load(os.path.join(basePath, xValidationDir))
        yValidation = np.load(os.path.join(basePath, yValidationDir))

        # fix para xTrain y xTrainValidation, para que estén en formato de canal
        xTrain = xTrain.reshape(xTrain.shape[0], 1, xTrain.shape[1])
        xValidation = xValidation.reshape(xValidation.shape[0], 1, xValidation.shape[1])
        # promediamos cada vector en yTrain e yValidation
        yTrain = np.array([np.array(y).mean() for y in yTrain])
        yValidation = np.array([np.array(y).mean() for y in yValidation])

        trainData = [xTrain, yTrain]
        validationData = [xValidation, yValidation]
        testData = [[], []]

        return trainData, validationData, testData
    return inner_mondayTuesday

def loadFinancialData_week_avgNext():
    xTrainDir = "../Datos/weeks_avgNext_2014_s1.npy"
    yTrainDir = "../Datos/weeks_avgNext_2014_s1_y.npy"
    xValidationDir = "../Datos/weeks_avgNext_2015_s1.npy"
    yValidationDir = "../Datos/weeks_avgNext_2015_s1_y.npy"

    xTrain = np.load(os.path.join(basePath, xTrainDir))
    yTrain = np.load(os.path.join(basePath, yTrainDir))

    xValidation = np.load(os.path.join(basePath, xValidationDir))
    yValidation = np.load(os.path.join(basePath, yValidationDir))

    # fix para xTrain y xTrainValidation, para que estén en formato de canal
    xTrain = xTrain.reshape(xTrain.shape[0], 1, xTrain.shape[1])
    xValidation = xValidation.reshape(xValidation.shape[0], 1, xValidation.shape[1])


    trainData = [xTrain[:40], yTrain[:40]]
    validationData = [xValidation, yValidation]
    testData = [[], []]

    return trainData, validationData, testData

x = loadFinancialData_mondayTuesday("wednesday")

