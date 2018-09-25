
import sys  # This is built into python
import csv  # Also built in to pyhthon
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from collections import Counter
from scipy import stats


def findLabel(TrainingData):

    LabelOutput = []
    for BoardState in TrainingData:
        if BoardState[42] == 1:
            LabelOutput.append(1)
        elif BoardState[42] == 2:
            LabelOutput.append(2)
        else:
            LabelOutput.append(0)

    return LabelOutput

def findBottomLeft(TrainingData):

    BottomLeftOutput = []
    for BoardState in TrainingData:
        if BoardState[0] == 1:
            BottomLeftOutput.append(1)
        elif BoardState[0] == 2:
            BottomLeftOutput.append(2)
        else:
            BottomLeftOutput.append(0)

    return BottomLeftOutput

def findMiddleColumn(TrainingData):

    MiddleColumnOutput = []
    Player1Counter = 0
    Player2Counter = 0;
    for BoardState in TrainingData:
        for Position in BoardState[18:24]:
            if Position == 1:
                Player1Counter += 1
            elif Position == 2:
                Player2Counter += 1
        if Player1Counter > Player2Counter:
            MiddleColumnOutput.append(1)
        elif Player1Counter < Player2Counter:
            MiddleColumnOutput.append(2)
        else:
            MiddleColumnOutput.append(0)
        Player1Counter = Player2Counter = 0

    return MiddleColumnOutput

def findBottomMiddle(TrainingData):

    BottomMiddleOutput = []
    for BoardState in TrainingData:
        if BoardState[18] == 1:
            BottomMiddleOutput.append(1)
        elif BoardState[18] == 2:
            BottomMiddleOutput.append(2)
        else:
            BottomMiddleOutput.append(0)

    return BottomMiddleOutput

def findMiddleMatrix(TrainingData):
    MiddleMatrixOutput = []
    Player1Counter = 0
    Player2Counter = 0
    for BoardState in TrainingData:
        for Position in range(16):
            if Position != 4:
                if Position != 5:
                    if Position != 10:
                        if Position != 11:
                            if BoardState[13 + Position] == 1:
                                Player1Counter += 1
                            elif BoardState[13 + Position] == 2:
                                Player2Counter += 1
        if Player1Counter > Player2Counter:
            MiddleMatrixOutput.append(1)
        elif Player1Counter < Player2Counter:
            MiddleMatrixOutput.append(2)
        else:
            MiddleMatrixOutput.append(0)
        Player1Counter = Player2Counter = 0

    return MiddleMatrixOutput

def findBottomRow(TrainingData):

    BottomOutput = []
    for BoardState in TrainingData:
        if ((BoardState[12] == BoardState[18] == BoardState[24]) and (BoardState[12] != 0)):
            BottomOutput.append(BoardState[12])
        else:
            BottomOutput.append(0)
    return BottomOutput

def KFoldDecisionTree(Fold,Data,Label,Criterion):
    # 3 fold validation with randomization
    equal = []
    KFoldValidation = KFold(n_splits=Fold,shuffle=True)
    for train_index, test_index in KFoldValidation.split(Data):
        TrainData, ValidationData = Data[train_index], Data[test_index]
        TrainLabel, ValidationLabel = Label[train_index], Label[test_index]

        DT = tree.DecisionTreeClassifier(criterion=Criterion)
        DT = DT.fit(TrainData, TrainLabel)
        Prediction = np.array(DT.predict(ValidationData))
        VL = np.array(ValidationLabel)
        equal = np.equal(Prediction,VL)
    return equal

def Statistic(Result,Data):
    for r in range(len(Result)):
        TrueCounter = Counter()
        for t in Result[r]:
            TrueCounter[t] += 1
        Accuracy = TrueCounter.most_common(1)[0][1] / (len(Data.transpose()) / Fold)
        AttriName = AttriNames[r]
        print("The Attribute is: ", AttriName)
        print("The Accuracy of this Test is: ", Accuracy)
        #get the rest of results for ttest
        RestOfResult = np.delete(np.array(Result),r,0)
        for r1 in RestOfResult:
            t = stats.ttest_ind(Result[r],r1)
            print("The ttests of this attribute set comparing to other attributes are: ", t)
        print('\n')


if __name__ == "__main__":

    # The *.csv files
    first_input_csv = sys.argv[2]
    second_output_csv = sys.argv[3]  # Same here

    # Now we want to open the *.csv file.
    #inputcsv = open(first_input_csv, 'r')  # This is means we are openning a file and then editing it.
    #TrainingCSV = csv.reader(inputcsv)  # Now we can read the file that was opened by python
    TrainingSet = pd.read_csv(first_input_csv, header=0)
    FindLabel = findLabel(TrainingSet.values)
    BottomLeft = findBottomLeft(TrainingSet.values)
    MiddleColumn = findMiddleColumn(TrainingSet.values)
    BottomMiddle = findBottomMiddle(TrainingSet.values)
    MiddleMatrix = findMiddleMatrix(TrainingSet.values)
    BottomRow = findBottomRow(TrainingSet.values)

    #all attribute data
    FullAttriData = np.array([BottomLeft,MiddleColumn,BottomMiddle,MiddleMatrix,BottomRow])
    #data with only the two required attributes
    RequiredAttriData = np.array([BottomLeft,MiddleColumn])
    #data with three added Attributes
    AddedAttriData = np.array([BottomMiddle,MiddleMatrix,BottomRow])

    #Labels
    Label = np.array(FindLabel)

    #TrainData, TestData, TrainLabel, TestLabel = train_test_split(Data, Label, test_size=0.1, random_state=100)

    #Testing Attribute effectiveness with method "Add One In" which include 2 of the required attributes and 1 of the added ones
    print('Testing Attribute effectiveness with method "Add One In"')
    print('The most effective attribute will have the best accuracy: ','\n')
    Result = []
    Fold = 3
    AttriNames = ['BottomMiddle','MiddleMatrix','BottomRow']
    for i in range(len(AddedAttriData)):
        #build data set
        Data = np.vstack([RequiredAttriData,AddedAttriData[i]])
        Data = Data.transpose()
        #initialize accuracy
        Accuracy = 0
        Result.append([])
        #build k fold decision tree and its prediction result
        Result[i] = KFoldDecisionTree(Fold,Data,Label,"entropy")

    #print out statics and tests on result
    Statistic(Result,RequiredAttriData)

    #Testing Attribute effectiveness with method "Leave One Out" which include 2 of the required attributes and 2 of the added ones
    print('Testing Attribute effectiveness with method "Leave One Out"')
    print('The most effective attribute will have the lowest accuracy due to its missing: ','\n')
    Result = []
    Fold = 3
    AttriNames = ['BottomMiddle','MiddleMatrix','BottomRow']
    for i in range(len(AddedAttriData)):
        #build data set
        Data = np.delete(FullAttriData,AddedAttriData[i],0)
        Data = Data.transpose()
        #initialize accuracy
        Accuracy = 0
        Result.append([])
        #build k fold decision tree and its prediction result
        Result[i] = KFoldDecisionTree(Fold,Data,Label,"entropy")

    #print out statics and tests on result
    Statistic(Result,RequiredAttriData)

    #Testing Attribute effectiveness with different best attribute selection method
    print('Testing Attribute effectiveness with different best attribute selection method')
    Result = []
    Fold = 3
    AttriNames = ['All','All']
    Data = FullAttriData.transpose()
    #initialize accuracy
    Accuracy = 0
    Result.append([])
    print('The method used in first test is Information Gain: ','\n')
    #build k fold decision tree and its prediction result
    Result[0] = KFoldDecisionTree(Fold,Data,Label,"entropy")

    Result.append([])
    print('The method used in second test is Gini Impurity: ','\n')
    #build k fold decision tree and its prediction result
    Result[1] = KFoldDecisionTree(Fold,Data,Label,"gini")

    #print out statics and tests on result
    Statistic(Result,RequiredAttriData)


    # Here are some changes we are going to make/create
    TrainingSet['Bottom Left'] = BottomLeft
    TrainingSet['Middle Column'] = MiddleColumn
    TrainingSet['Bottom Middle'] = BottomMiddle
    TrainingSet['Middle Matrix'] = MiddleMatrix
    TrainingSet['Bottom Row'] = BottomRow

    print(TrainingSet.head())

    """
    Now we will write the changes to the output file.

    First, we need to take in the string that is the 
    filename and create a filepath out of it.

    """

    # Now, if the rejectme.csv file is being viewed in the GUI
    # before the script runs, you will get write permission errors.

    # Creating the path
    TrainingSet.to_csv(str(second_output_csv))

    """
    Now you have read in and written to a *.csv file
    the way the graders will.

    """
