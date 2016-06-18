from NeuralPython.Utils import Builders
from Utils import LoadData
import json
import sys

yDay = sys.argv[1]
print yDay

dirs = ["1466105328/", "1466105331/", "1466105334/"]
if yDay == "wednesday":
    networkDir = dirs[0]
elif yDay == "thursday":
    networkDir = dirs[1]
elif yDay == "friday":
    networkDir = dirs[2]

loadDataFunc = LoadData.loadFinancialData_mondayTuesday(yDay)
config = json.load(open("Config_mondayTuesday.json"))
config["networkLoadDir"] = networkDir
net = Builders.buildNetwork(config)
mpiTrain = Builders.buildTraining(config)
mpiTrain.setNetwork(net)
mpiTrain.loadData(loadDataFunc)
mpiTrain.run()
