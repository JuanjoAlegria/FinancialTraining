import os
import numpy as np

def loadFinancialData():
    basePath = os.path.dirname(os.path.realpath(__file__))
    xTrainDir = "../Datos/x2014_both.npy"
    yTrainDir = "../Datos/y2014_avg_delta.npy"
    xValidationDir = "../Datos/x2015_both.npy"
    yValidationDir = "../Datos/y2015_avg_delta.npy"

    xTrain = np.load(os.path.join(basePath, xTrainDir))
    yTrain = np.load(os.path.join(basePath, yTrainDir))

    xValidation = np.load(os.path.join(basePath, xValidationDir))
    yValidation = np.load(os.path.join(basePath, yValidationDir))

    trainData = [xTrain, yTrain]
    validationData = [xValidation[:35], yValidation[:35]]
    testData = [[], []]

    return trainData, validationData, testData
