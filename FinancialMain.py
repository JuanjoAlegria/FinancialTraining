from NeuralPython.Utils import Builders
from Utils import LoadData
import json

config = json.load(open("Config.json"))
net = Builders.buildNetwork(config)
mpiTrain = Builders.buildTraining(config)
mpiTrain.setNetwork(net)
mpiTrain.loadData(LoadData.loadFinancialData)
mpiTrain.run()
