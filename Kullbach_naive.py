import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import sys
from scipy.io import arff
import pandas as pd
import math
data = arff.loadarff('training.arff')
df = pd.DataFrame(data[0])
data = arff.loadarff('testing.arff')
df2 = pd.DataFrame(data[0])

def naive(weights):
    X = np.loadtxt(open("final_dataset_genetic.csv", "rb"), delimiter=",", usecols=range(14), skiprows=1, dtype=int)
    y = np.loadtxt(open("final_dataset_genetic.csv", "rb"), delimiter=",", usecols=range(14,15), skiprows=1, dtype=int)

    Xd_train, Xd_test, y_train, y_test = train_test_split(X, y, random_state = 0, stratify = y, test_size=0.3)


    num_observations = dict()
    for num in set(y):
        num_observations[num] = list(y_train).count(num)


    # print(f"\n# in each class value bucket: {num_observations}")
    training_num_observations = sum(num_observations.values())
    # print(f"# of total instances in the training set: {training_num_observations}\n")
    probabilities = dict()


    # print("Training...")
    # print("Calculating all probabilities to use during testing...")

    for num in set(y): # num = equal a class value 
        probabilities[num] = dict() # establishes a dictionary for each class value 
        for i in range(len(X[0])): # now looping through each of the attribute / features 
            probabilities[num][i] = dict() # establishes a dictionary for each attribute
            for j in range(training_num_observations): # now looping through each of the observations
                if y_train[j] == num: # if the observation has the same class value
                    if Xd_train[j][i] not in probabilities[num][i]: # let's check if the attribute value is in the dictionary
                        probabilities[num][i][Xd_train[j][i]] = 0 # if not, let's add it and set it to 0
                    probabilities[num][i][Xd_train[j][i]] += 1 # now we increment the value of the attribute by 1
            for value in range(6): 
                if value not in probabilities[num][i]: # if the attribute value is not in the dictionary
                    probabilities[num][i][value] = 0 # let's add it and set it to 0
            probabilities[num][i] = {k: v / num_observations[num] for k, v in probabilities[num][i].items()} # now we divide the value of the attribute by the number of observations of that class value


    # print("Probabilities calculated!\n")


    # print(f"\n# of test observations: {len(Xd_test)}")
    # print("Testing...\n")

    correct = 0
    total = 0
    confusion_matrix = np.zeros((len(set(y)), len(set(y))), dtype=int)
    for i in range(len(Xd_test)):
        testing_now = Xd_test[i]
        probs = []
        for num in set(y):
            accumulator = 1
            for attribute_value in range(len(testing_now)):
                accumulator = accumulator * (probabilities[num][attribute_value][testing_now[attribute_value]]**weights[attribute_value])
            accumulator = accumulator * (num_observations[num] / training_num_observations)
            probs.append(accumulator)
        prediction = probs.index(max(probs))
        if prediction == y_test[i]:
            correct += 1
        total += 1
        confusion_matrix[y_test[i],prediction] = confusion_matrix[y_test[i],prediction] + 1
        # print("i:", i, "Prediction: ", prediction, "Actual: ", y_test[i], "Correct: ", prediction == y_test[i])

    # print("\nTesting accuracy", correct/total * 100, "%")

    return correct/total * 100


def conditionalProb(df,attribute,class_label,attribute_label, key=-1):
    numerator = 1
    denominator = 0
    for x in range(len(df)):
        if df[df.keys()[key]][x] == class_label:
            denominator += 1
            if df[attribute][x] == attribute_label:
                numerator += 1
    if denominator == 0:
        return 0.0
    return numerator/denominator

def prop(df,attribute,val):
    numerator = 1
    denominator = 0
    for x in range(len(df)):
        denominator += 1
        if df[attribute][x] == val:
            numerator += 1
    if denominator == 0:
        return 0.0
    return numerator/denominator

def prob(att1,att1val,att2,att2val):
    num = 1
    den = 1
    for x in range(len(df)):
        if df[att2][x] == att2val:
            den += 1
            if df[att1][x] == att1val:
                num += 1
    return num/den

def kullbach_leibler(classes, attribute, att_label):
    kullbach = 0
    for x in classes:
        temp = prob(df.keys()[-1],x,attribute,att_label)
        temp = temp*math.log(temp/prop(df,df.keys()[-1],x))
        kullbach += temp
    return kullbach
    
def weighted_average(df,attribute,atts,classes):
    weight = 0
    for x in atts:
        weight += prop(df,attribute,x)*kullbach_leibler(classes,attribute,x)
    return weight

def split_info(df,attribute,atts):
    info = 0
    for x in atts:
        info += prop(df,attribute,x)*math.log(prop(df,attribute,x))
    return -info

def calc_Z(n,weights):
    z = 1/n
    sum = 0
    for x in range(n):
        sum += weights[x]
    z = z*sum
    return z

def sigmoid(num):
    return 1/(1+math.e**(-num))

def calc_weights(df,atts,classes):
    weights = []
    for x in range(len(atts)):
        weights.append(weighted_average(df,df.keys()[x],atts[x],classes)/split_info(df,df.keys()[x],atts[x]))
    for q in range(1000):
        z = calc_Z(len(atts),weights)
        for x in range(len(weights)):
            weights[x] = (1/z)*weights[x]
    for x in range(len(weights)):
        weights[x] = sigmoid(weights[x])
    return weights


classes = []
for x in range(len(df)):
    classes.append(df[df.keys()[-1]][x])
atts = []
for x in df.keys():
    if not x == df.keys()[-1]:
        temp = []
        for y in range(len(df)):
            temp.append(df[x][y])
        temp = set(temp)
        temp = list(temp)
        atts.append(temp)
classes = set(classes)
classes = list(classes)
print(naive(calc_weights(df,atts,classes)))




