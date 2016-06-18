from NeuralPython.Utils import Builders
from Utils import LoadData
import json

config = json.load(open("Config_week_to_week.json"))
config["networkLoadDir"] = "1466280861/"
net = Builders.buildNetwork(config)
mpiTrain = Builders.buildTraining(config)
mpiTrain.setNetwork(net)
mpiTrain.loadData(LoadData.loadFinancialData_week_avgNext)
mpiTrain.run()
