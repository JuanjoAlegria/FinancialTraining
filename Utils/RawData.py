# -*- coding: utf-8 -*-
import os
import csv
import datetime
import numpy as np

baseDir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "../Datos/data.2015.11.26/ticks/EURUSD-{0:04d}-{1:02d}.csv")


def loadMonthData(year, month, interval):
    csvFile = open(baseDir.format(year, month), 'r')
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    currentDay = datetime.datetime(year, month, 1)
    delta = datetime.timedelta(0, interval)

    times = []
    s1 = []
    s2 = []

    accumulatedS1 = 0
    accumulatedS2 = 0
    nOfTimes = 0

    for row in reader:
        # nos aseguramos que estemos en un día de semana
        while currentDay.isoweekday() in [6, 7]:
            currentDay += delta

        time = datetime.datetime(*timeStringToTuple(row[1]))
        if time < currentDay + delta:
            accumulatedS1 += float(row[2])
            accumulatedS2 += float(row[3])
            nOfTimes += 1
            continue

        if nOfTimes != 0:
            times.append(currentDay)
            s1.append(accumulatedS1 / nOfTimes)
            s2.append(accumulatedS2 / nOfTimes)
        accumulatedS1 = 0
        accumulatedS2 = 0
        nOfTimes = 0
        currentDay += delta

    if nOfTimes != 0:
        times.append(currentDay)
        s1.append(accumulatedS1 / nOfTimes)
        s2.append(accumulatedS2 / nOfTimes)

    return times, s1, s2



def load(year1, month1, year2, month2, dataInterval):
    dataInterval *= 60 # conversión a segundos
    times = []
    serie1 = []
    serie2 = []
    tuplesYearsMonth = buildYearMonthTuples(year1, month1, year2, month2)
    for y, m in tuplesYearsMonth:
        t, s1, s2 = loadMonthData(y, m, dataInterval)
        times += t
        serie1 += s1
        serie2 += s2
    return times, serie1, serie2


def buildYearMonthTuples(year1, month1, year2, month2):
    """
        Return a list of tuples of the form (year, month) for all the
        year-month pairs between year1, month1 to year2, month2
        (both inclusive)
    """
    tuples = []
    for y in range(year1, year2 + 1):
        initialMonth = 1 if y != year1 else month1
        lastMonth = 12 if y != year2 else month2
        tuples += [(y, m) for m in range(initialMonth, lastMonth + 1)]
    return tuples

def timeStringToTuple(s):
    """
        Return year, month, day, hour, minute, second from format
        YYMMDD hh:mm:ss
    """
    sTuple = (s[:4], s[4: 6], s[6: 8], s[9:11], s[12:14], s[15:17])
    return map(int, sTuple)


if __name__ == "__main__":
    [time, s1, s2]  = load(2014, 1, 2014, 12, 1440)
    print "###################################"
    print np.shape(time)
    print np.shape(s1)
    print np.shape(s2)
    print time[0]
    print time[-1]
