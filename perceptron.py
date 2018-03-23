import csv
import os
import sys
import numpy as np
import random as rand
import math
import copy as cp
from PIL import Image

class Neuron():
    def __init__(self,w,ww,lrate):
        self.w0 = w
        self.wn = np.array(ww)
        self.learning_rate = lrate

    def getw(self):
        return self.w0, list(self.wn)

    def training(self,x, w = None, ww = None):
        if w != None and ww != None:
            self.w0 = w
            self.wn = ww
        out = []
        target = []
        g = lambda x: 1 if x > 0 else -1
        for e in x:
            t = e[-1]
            target.append(t)
            a = np.array(list(map(lambda x: float(x),e[:-1])))
            o = self.w0 + a.dot(self.wn)
            out.append(o[0])
            self.learning(int(t),out[-1],a)
        return out

    def learning(self,t,o,x):
        self.w0 = [self.w0[0] + (self.learning_rate*(t-o))]
        for i in range(0,len(x)):
            self.wn[i] = self.wn[i] + (self.learning_rate*(t-o)*x[i])

    def classify(self,x,w = None,ww = None):
        if w != None and ww != None:
            self.w0 = w
            self.wn = ww
        out = []
        target = []
        g = lambda x: 1 if x > 0 else -1
        for e in x:
            t = e[-1]
            target.append(int(t))
            a = np.array(list(map(lambda x: float(x),e[:-1])))
            o = self.w0 + a.dot(self.wn)
            out.append(g(o))
        return out, target


fil = csv.reader(open('/home/ld/Desktop/occupancy.csv', "rt"))
dataset = np.array(list(fil))
test_rate = 0.99
rand_set = rand.sample(range(0,len(dataset)),math.floor(test_rate*len(dataset)))
test = dataset[rand_set]
train = cp.deepcopy(dataset.tolist())
for i in rand_set:
    train[i] = 0
train = np.array(list(filter(lambda x: x != 0, train)))
print([len(train),len(test)])

accuracy = 0
counts = 1
a = list(map(lambda x: x/10,rand.sample(range(-9,9),1)))
b = list(map(lambda x: x/10,rand.sample(range(-9,9),5)))
n = Neuron(a,b,0.000001)
n.training(train)
w0, wn = n.getw()
out, target = n.classify(test)
acc = []
for x,y in zip(out,target):
    acc.append(x==y)
accuracy = sum(acc)*100/len(acc)
while accuracy < 79:
    n.training(train,w0,wn)
    w0, wn = n.getw()
    out, target = n.classify(test)
    # print(n.wn)
    # print(n.w0)
    acc = []
    for x,y in zip(out,target):
        #print([x,y])
        acc.append(x==y)
    accuracy = sum(acc)*100/len(acc)
    counts = counts + 1
print([accuracy,n.getw()])
print(counts)


# n = Neuron([0.1],list(map(lambda x: x/10,rand.sample(range(-9,9),5))),0.00001)
# out, target = n.classify(test,[-0.49597623725694934],[-0.67061476, -0.08122922,  0.03820375,  0.00581532,  0.69997643])
# acc = []
# for x,y in zip(out,target):
#     #print([x,y])
#     acc.append(x==y)
# accuracy = sum(acc)*100/len(acc)
# print(accuracy)
