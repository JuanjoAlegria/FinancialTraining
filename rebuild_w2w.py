# -*- coding: utf-8 -*-
import json
from Utils import LoadData
from NeuralPython.Utils import NaiveError, Rebuild
train, validation, test = LoadData.loadFinancialData_week_avgNext()
import pdb; pdb.set_trace()  # breakpoint a0539c34 //

outputNaiveTrain, c1NaiveTrain, c2NaiveTrain = NaiveError.naiveAlgorithm(train)
outputNaiveVal, c1NaiveVal, c2NaiveVal = NaiveError.naiveAlgorithm(validation)

print c1NaiveTrain, c2NaiveTrain
print c1NaiveVal, c2NaiveVal

#dirs = ["1466144331/"] #12000 epochs, eta = 0.00001
dirs = ["1466144331/", "1466149884/", "1466274918/", "1466280861/", "1466282019/"] #casi 100000 epochs, 10 epochs más
config = json.load(open("Config_week_to_week.json"))

for i in range(len(dirs)):
    print "######################################################"
    currentDirs = [dirs[i], dirs[i] + "BestResult/"]
    for dataset, datasetName in [(train, "Entrenamiento"), (validation, "Validación")]:
        outputs, c1, c2 = Rebuild.rebuildSignal(currentDirs, dataset, config)
        print "Conjunto", datasetName
        print "Error C1 entrenamiento exhaustivo: ", c1[0][0]
        print "Error C2 entrenamiento exhaustivo: ", c2[0][0]
        print "Error C1 mejor resultado: ", c1[1][0]
        print "Error C2 mejor resultado: ", c2[1][0]
        Rebuild.plotSignal(dataset, outputs)

