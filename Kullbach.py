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
    for q in range(100):
        z = calc_Z(len(atts),weights)
        for x in range(len(weights)):
            weights[x] = (1/z)*weights[x]
    for x in range(len(weights)):
        weights[x] = sigmoid(weights[x])
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
    print(weights)
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