import numpy as np
import json
from NeuralPython.Utils import Rebuild
from Utils import LoadData

dirs = ["1465466957/", "1465467257/", "1465471722/"]
errors = [0.001, 0.00001, 0.000001]
config = json.load(open("Config.json"))
train, validation, test = LoadData.loadFinancialData()
outputs, c1, c2 = Rebuild.rebuildSignal(dirs, validation, errors, config)

print c1
print c2

total = []
for i in range(len(validation[0])):
    x = validation[0][i]
    avg = (x[0][-1] + x[1][-1]) / 2
    validation[1][i] += avg
    delta = outputs[2][i]
    total.append(avg + delta)


total = np.array(total)

Rebuild.plotSignal(validation, [total])
