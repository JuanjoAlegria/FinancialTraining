import numpy as np
import RawData


def returnOnInvestment(cost, benefit):
    return np.log(benefit / cost) * 100


def yInterval(cost, benefit, intervals):
    r = returnOnInvestment(cost, benefit)

    for i in range(len(intervals) - 1):
        if intervals[i] <= r < intervals[i + 1]:
            return i
    raise Exception("r = " + str(r))


def fixIntervals(intervals, zeroEpsilon=0.00001):
    intervalsCopy = intervals[:]
    if 0 in intervalsCopy:
        zeroIndex = intervalsCopy.index(0)
        intervalsCopy[zeroIndex] = -zeroEpsilon
        intervalsCopy.insert(zeroIndex + 1, zeroEpsilon)
    intervalsCopy.insert(0, float('-inf'))
    intervalsCopy.append(float('inf'))
    return intervalsCopy


def buildXExtra(date):
    month = date.month
    day = date.day
    weekDay = date.isoweekday()
    return np.array((month, day, weekDay))


def vectorize(series, times, regHorizon, predHorizon, yRanges, desiredSerieIndex=0):
    assert len(times) == len(series[0])
    lenVectors = len(times)
    xTotal = []
    xExtraTotal = []
    yTotal = []

    for i in range(regHorizon, lenVectors - predHorizon):
        # construimos x = list(float), shape = [regHorizon,]
        x = []
        for serie in series:
            x.append(np.array(serie[i - regHorizon: i]))
        # construimos xExtra = np.array(int), shape = [4,]
        xExtra = buildXExtra(times[i])
        # construimos y = float
        y = returnOnInvestment(series[desiredSerieIndex][i],
                               series[desiredSerieIndex][i + predHorizon])
        # y = returnOnInvestment(series[desiredSerieIndex][i],
        #               series[desiredSerieIndex][i + predHorizon], yRanges)
        xTotal.append(x)
        xExtraTotal.append(xExtra)
        yTotal.append(y)

    # np.savetxt("x.csv", xTotal, fmt="%s", delimiter = ",")
    # np.savetxt("xExtra.csv", xExtra,  fmt="%s", delimiter = ",")
    return xTotal, xExtraTotal, yTotal


def permute(*arrays):
    result = []
    perm = np.random.permutation(len(arrays[0]))
    for a in arrays:
        if a == []:
            result.append(a)
        else:
            result.append(a[perm])
    return tuple(result)


def load(y1, m1, y2, m2, regHorizon, predHorizon, yRanges, dataInterval):
    [times, serie1, serie2] = RawData.load(y1, m1, y2, m2, dataInterval)
    x, xExtra, y = vectorize((serie1,), times,
                             regHorizon, predHorizon, yRanges)
    x = np.array(x)
    xExtra = np.array(xExtra).astype(float)
    y = np.array(y).astype(float)

    return x, xExtra, y


def loadTest(y1, m1, y2, m2, regHorizon, predHorizon, intervals,
             dataInterval, nTest):
    x, xExtra, y = load(
        y1, m1, y2, m2, regHorizon, predHorizon, intervals, dataInterval)
    x, xExtra, y = permute(x, xExtra, y)
    return x[:nTest], xExtra[:nTest], y[:nTest]


def loadFullWeeks(year1, month1, year2, month2, dataInterval, indexSerie = 0):
    weeks = []
    series = []
    rawSeries = []
    times, s0, s1 = RawData.load(year1, month1, year2, month2, dataInterval)
    rawSeries.append(s0)
    rawSeries.append(s1)

    currentWeek = []
    currentS = []
    currentWeekNumber = times[0].isocalendar()[1]

    for t, s in zip(times, rawSeries[indexSerie]):
        if t.isocalendar()[1] != currentWeekNumber:
            weeks.append(currentWeek)
            series.append(currentS)
            currentWeekNumber = t.isocalendar()[1]
            currentWeek = []
            currentS = []
        currentWeek.append(t)
        currentS.append(s)

    if currentWeek not in weeks:
        weeks.append(currentWeek)
        series.append(currentS)

    return postLoad(weeks, [series])


def avg_tuesday(week, serie):
    tuesday = []
    i = 24
    while True:
        if week[i].isoweekday() == 2:
            i += 1
        else:
            break

    tuesday = serie[24: i]
    return 1.0 * sum(tuesday) / len(tuesday)


def postLoad(weeks, series):

    wReturn = []
    sReturn = []
    yReturn = []
    tuesdayReturn = []

    for i in range(len(weeks) - 1):
        w = weeks[i]
        w_next = weeks[i + 1]
        if len(w) >= 117 and (w_next[0].isoweekday(), w_next[0].hour) == (1, 0) and (w_next[23].isoweekday(), w_next[23].hour) == (1, 23) and w_next[24].isoweekday() == 2:
            w = w[:117]
            w += w_next[:24]
            tuesdayReturn.append(w_next[24])
            s = []

            for j in range(len(series)):
                current_s = series[j][i]
                s_next = series[j][i + 1]
                current_s = current_s[:117]
                current_s += s_next[:24]
                s.append(np.array(current_s))

            wReturn.append(np.array(w))
            sReturn.append(np.array(s))
            yReturn.append(avg_tuesday(w_next, s_next))

    return np.array(wReturn), np.array(tuesdayReturn), np.array(sReturn), np.array(yReturn)

def loadFromDict(d):
    return load(d['year1Train'], d['month1Train'], d['year2Train'], d['month2Train'],
                d['regHorizon'], d['predHorizon'], d['yRanges'], d['dataInterval'])


def loadTestFromDict(d):
    return loadTest(d['year1Test'], d['month1Test'], d['year2Test'],
                    d['month2Test'], d['regHorizon'], d['predHorizon'],
                    d['yRanges'], d['dataInterval'],
                    d['miniBatchTestSize'] * d['nProcesses'])


def loadAndSaveFullYears(year, firstMonth = 1, lastMonth = 12):
    w1, t1, s1, y1 = loadFullWeeks(year, firstMonth, year, lastMonth, 60, \
                                   indexSerie = 0)
    w2, t2, s2, y2 = loadFullWeeks(year, firstMonth, year, lastMonth, 60, \
                                   indexSerie = 1)
    if (w1 == w2).all() and (t1 == t2).all():
        np.save("x" + str(year) + "_s1.npy", s1)
        np.save("x" + str(year) + "_s2.npy", s2)
        np.save("y" + str(year) + "_s1.npy", y1)
        np.save("y" + str(year) + "_s2.npy", y2)
        np.save("x" + str(year) + "_avg.npy", (s1 + s2) / 2)
        np.save("y" + str(year) + "_avg.npy", (y1 + y2) / 2)
        np.save("weeks" + str(year) + ".npy", w1)
        np.save("tuesdays" + str(year) + ".npy", t1)

# for year in [2014, 2015]:
#     for suffix in ['_s1', '_s2', '_avg']:
#         x_filename = "x" + str(year) + suffix + ".npy"
#         y_filename = "y" + str(year) + suffix + ".npy"

#         x = np.load(x_filename)
#         y = np.load(y_filename)

#         last_x = np.array([x[i][0][-1] for i in range(len(x))])
#         delta = y - last_x

#         y_delta_filename = "y" + str(year) + suffix + "_delta.npy"
#         np.save(y_delta_filename, delta)

# for year in [2014, 2015]:
#     x1_name = "x" + str(year) + "_s1.npy"
#     x2_name = "x" + str(year) + "_s2.npy"
#     x1 = np.load(x1_name)
#     x2 = np.load(x2_name)
#     xTotal = np.concatenate((x1, x2), axis = 1)
#     np.save("x" + str(year) + "_both.npy", xTotal)
