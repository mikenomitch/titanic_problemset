import numpy
import pandas
import math
import statsmodels.api as sm

# NOTE: This code is pretty ugly to be honest
# but I included it to supplement the answer to
# question #4

def custom_heuristic(file_path):
    predictions = {}
    df = pandas.read_csv(file_path)
    columnsICareAbout = ['Pclass', 'Sex', 'Age', 'Parch', 'SibSp']

    # for each column turn data into something
    # that we can take ave of survival rate for those vals
    lambdasForColumn = {}
    lambdasForColumn['Sex'] = lambda x: x == 'male'
    lambdasForColumn['Age'] = lambda x: x < 16
    lambdasForColumn['Pclass'] = lambda x: x
    lambdasForColumn['Parch'] = lambda x: x > 2
    lambdasForColumn['SibSp'] = lambda x: x > 2
    lambdasForColumn['Embarked'] = lambda x: x == 'C'

    ratesForValuesInColums = {}
    firstColRates = {}
    secondCallRatesWithinFirst = {}
    rates = {}

    # create new cols for coerced vals
    new_cols = [str(x) + "_lam" for x in columnsICareAbout]
    for colName in columnsICareAbout:
        lambaForCol = lambdasForColumn[colName]
        df[colName + "_lam"] = df[colName].apply(lambaForCol)

    firstRunIdx = 1
    for firstColName in columnsICareAbout:
        ratesForValuesInColums[firstColName] = {}
        remainingColumns = columnsICareAbout[(firstRunIdx):]
        if remainingColumns:
            firstColRates = {}
            for secondColName in remainingColumns:
                secondCallRatesWithinFirst[secondColName] = {}

                firstModLambda = lambdasForColumn[firstColName]
                secondModLambda = lambdasForColumn[secondColName]
                firstColumnValues = df[firstColName].apply(firstModLambda)
                secondColumnValues = df[secondColName].apply(secondModLambda)

                def getRatesForValues(valA, valB):
                    matchingBoth = df.loc[df[firstColName+"_lam"] == valA].loc[df[secondColName+"_lam"] == valB]["Survived"]
                    survivalRate = matchingBoth.dropna().mean()
                    secondCallRatesWithinFirst[secondColName][valB] = matchingBoth.dropna().mean()
                    if not math.isnan(survivalRate):
                        likelyDead = survivalRate < 0.5
                        if likelyDead:
                            distanceFromHalf = 0.5 - survivalRate
                        else:
                            distanceFromHalf = survivalRate - 0.5

                        rates[distanceFromHalf] = {}
                        rates[distanceFromHalf]['survivalRate'] = survivalRate
                        rates[distanceFromHalf]['columnA'] = firstColName
                        rates[distanceFromHalf]['valA'] = valA
                        rates[distanceFromHalf]['lamA'] = firstModLambda
                        rates[distanceFromHalf]['columnB'] = secondColName
                        rates[distanceFromHalf]['valB'] = valB
                        rates[distanceFromHalf]['lamB'] = secondModLambda
                        rates[distanceFromHalf]['likelyDead'] = likelyDead
                        if likelyDead:
                            rates[distanceFromHalf]['prediction'] = 0
                        else:
                            rates[distanceFromHalf]['prediction'] = 1


                firstUniqValues = secondColumnValues.unique()
                secondUniqValues = firstColumnValues.unique()
                for valA in firstUniqValues:
                    for valB in secondUniqValues:
                        secondCallRatesWithinFirst[secondColName][valB] = {}
                        getRatesForValues(valA, valB)

                ratesForValuesInColums[firstColName][valA] = secondCallRatesWithinFirst
            firstRunIdx = firstRunIdx + 1

    def makeAGuess(passenger):
        print('STARTING STARTING STARTING STARTING STARTING STARTING')
        sortedDecisionKeys = reversed(sorted(rates.keys()))
        decision = 'undecided'
        for decisionKey in sortedDecisionKeys:
            if decision == 'undecided':
                colA = rates[decisionKey]['columnA']
                valA = rates[decisionKey]['valA']
                lamA = rates[decisionKey]['lamA']
                colB = rates[decisionKey]['columnB']
                valB = rates[decisionKey]['valB']
                lamB = rates[decisionKey]['lamB']

                if ((lamA(passenger[colA]) == valA) and (lamB(passenger[colB]) == valB)):
                    decision = rates[decisionKey]['prediction']

        if decision == 'undecided':
            decision = 0
        return decision

    for passenger_index, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']
        predictions[passenger_id] = makeAGuess(passenger)

    return predictions

custom_heuristic('./titanic_data.csv')
