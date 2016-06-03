import os
import json
import numpy as np
from NeuralPython import MPINetwork

def loadFinancialData():
    basePath = os.path.dirname(os.path.realpath(__file__))
    xTrainDir = "Datos/xTrain.npy"
    yTrainDir = "Datos/yTrain.npy"
    xValidationDir = "Datos/xTest.npy"
    yValidationDir = "Datos/yTest.npy"

    xTrain = np.load(os.path.join(basePath, xTrainDir))
    yTrain = np.load(os.path.join(basePath, yTrainDir))
    xValidation = np.load(os.path.join(basePath, xValidationDir))
    yValidation = np.load(os.path.join(basePath, yValidationDir))

    trainData = [xTrain, [], yTrain]
    validationData = [xValidation, [], yValidation]
    testData = [[], [], []]

    return trainData, validationData, testData

config = json.load(open("Config.json"))
mNet = MPINetwork.buildFromDict(config)
mNet.loadData(loadFinancialData)
mNet.run()

