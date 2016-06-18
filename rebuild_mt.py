# -*- coding: utf-8 -*-
import json
from NeuralPython.Utils import Rebuild, NaiveError
from Utils import LoadData

# dirs = ["1466097227/", "1466097224/", "1466097221/"] 300 epochs
# dirs = ["1466102111/", "1466102114/", "1466102117/"] 3000 epochs
dirs = ["1466105328/", "1466105331/", "1466105334/"]   # 10000 epochs
days = ["wednesday", "thursday", "friday"]
dias = ["Miércoles", "Jueves", "Viernes"]
config = json.load(open("Config_mondayTuesday.json"))

for i in range(3):
    print "######################################################"
    print dias[i]
    train, validation, test = LoadData.loadFinancialData_mondayTuesday(days[i])()
    dayDirs = [dirs[i], dirs[i] + "BestResult/"]
    for dataset, datasetName in [(train, "Entrenamiento"), (validation, "Validación")]:
        outputs, c1, c2 = Rebuild.rebuildSignal(dayDirs, dataset, config)
        print "Conjunto", datasetName
        print "Error C1 entrenamiento exhaustivo: ", c1[0][0]
        print "Error C2 entrenamiento exhaustivo: ", c2[0][0]
        print "Error C1 mejor resultado: ", c1[1][0]
        print "Error C2 mejor resultado: ", c2[1][0]
        Rebuild.plotSignal(dataset, outputs)

        outputNaive, c1Naive, c2Naive = NaiveError.naiveAlgorithm(dataset)
        Rebuild.plotSignal(dataset, [outputNaive])
        print "Error C1 algoritmo ingenuo", c1Naive
        print "Error C2 algoritmo ingenuo", c2Naive
        print "----------------------------------"
