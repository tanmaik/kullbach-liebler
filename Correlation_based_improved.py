from scipy.io import arff
import pandas as pd
import math
data = arff.loadarff('training.arff')
df = pd.DataFrame(data[0])
data = arff.loadarff('testing.arff')
df2 = pd.DataFrame(data[0])

def display_confusion(confusion):
   print('Order: ', end = '')
   for x in confusion:
      print(x, end = ' ')
   print()
   for x in confusion:
      print(confusion[x])

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

def create_instance(df,z):
    instance = []
    for x in df.keys():
        if not x == df.keys()[-1]:
            instance.append(df[x][z])
    return instance

def calc_prob(probs,instance,class_label,class_prob,weights):
    prob = class_prob/len(df)
    y = 0
    for x in instance:
        prob = prob*(probs[x][class_label]**weights[y])
        y += 1
    return prob

def class_probs(classes,df):
    thingy = {}
    for x in range(len(df)):
        if df[df.keys()[-1]][x] not in thingy:
            thingy[df[df.keys()[-1]][x]] = 1
        else:
            thingy[df[df.keys()[-1]][x]] = thingy[df[df.keys()[-1]][x]]+1
    return thingy

def log(condit, prob1,prob2):
    return math.log((condit/prob1*prob2))

def informationClass(df,attribute, labels, classes):
    toret = 0
    for x in labels:
        for y in classes:
            toret += conditionalProb(df, attribute, y, x)*log(conditionalProb(df,attribute,y,x),prop(df,attribute,x),prop(df,df.keys()[-1],y))
    return toret

def mutualinfo(df,att1,att2num,labels1,labels2):
    toret = 0
    for x in labels1:
        for y in labels2:
            toret += conditionalProb(df, att1, y, x, att2num)*log(conditionalProb(df,att1,y,x, att2num),prop(df,att1,x),prop(df,df.keys()[att2num],y))
    return toret

def NI(df,attribute, atts, attnum, classes):
    num = informationClass(df,attribute,atts[attnum],classes)
    den = 0
    for y in range(len(atts)):
        den += informationClass(df,df.keys()[y],atts[y],classes)
    den = (1/len(atts))*den
    return num/den

def NImut(df,att1,att1num,att2num, atts):
    m = len(atts)
    num = mutualinfo(df,att1,att2num,atts[att1num],atts[att2num])
    den = 0
    for x in range(len(atts)):
        for y in range(len(atts)):
            if not y == x:
                den += mutualinfo(df,df.keys()[x],y,atts[x],atts[y])
    return num/((1/(m*(m-1)))*den)

def sigmoid(num):
    return 1/(1+math.e**(num))

def calc_weights(df,atts,classes):
    weights = []
    m = len(atts)
    for x in range(m):
        weight = NI(df,df.keys()[x],atts,x,classes)
        helper = 0
        for y in range(m):
            if not y == x:
                helper += NImut(df,df.keys()[x],x,y,atts)
        helper = (1/(m-1))*helper
        weight = weight-helper
        weights.append(sigmoid(weight))
    return weights

def naive_bayes():
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
    class_dist = class_probs(classes,df)
    probs = {}
    holder = []
    class_number = {}
    for x in range(len(classes)):
        class_number[classes[x]] = x
        holder.append(1/len(df))
    for x in range(len(df)):
        for y in df.keys():
            if not df[y][x] == df[df.keys()[-1]][x]:
                if df[y][x] not in probs:
                    probs[df[y][x]] = holder[:]
                if probs[df[y][x]][class_number[df[df.keys()[-1]][x]]] == 1/len(df):
                    probs[df[y][x]][class_number[df[df.keys()[-1]][x]]] = conditionalProb(df,y,df[df.keys()[-1]][x],df[y][x])
    correct = 0
    holder = []
    confusion = {}
    for x in range(len(classes)):
        holder.append(0)
    for x in classes:
        confusion[x] = holder[:]
    weights = calc_weights(df,atts,classes)
    for x in range(len(df2)):
        max = 0
        classification = 0
        instance = create_instance(df2,x)
        for y in range(len(classes)):
            probability = calc_prob(probs,instance,y,class_dist[classes[y]],weights)
            if probability > max:
                max = probability
                classification = y
        confusion[df2[df2.keys()[-1]][x]][class_number[classes[classification]]] += 1
        if classes[classification] == df2[df2.keys()[-1]][x]:
            correct += 1
    print("NaiveBayes Accuracy: " + str(correct/len(df2)*100) + '%')
    display_confusion(confusion)
    print()


def main():
    naive_bayes()

if __name__ == '__main__': main()