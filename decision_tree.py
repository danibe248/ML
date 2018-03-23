import csv
import os
import sys
import numpy as ny
import random as rand
import math
import copy as cp
from PIL import Image

class Tree():
    def __init__(self):
        self.name = None
        self.sons = {}

def entropy(f):
    d = []
    for row in f:
        d.append(row[-1])
    a = list(set(d))
    e = 0
    for s in a:
        r = d.count(s)/len(d)
        e = e-r*ny.log2(r)
    return(e)


def gain(f,a):
    cs = []
    v = []
    for row in f:
        v.append(row[a])
    supp = list(set(v))
    sumparz = 0
    # iv = 0
    for s in supp:
        z = [i for i, x in enumerate(v) if x==s]
        cs=list(f[z])
        sumparz = sumparz + (v.count(s)/len(v))*entropy(cs)
        # iv  = iv + (v.count(s)/len(v))*math.log2(v.count(s)/len(v))
    return entropy(f) - sumparz
    # return (entropy(f)-sumparz)/(-iv)


def dtl(examples,attributes,parent_examples):
    cs = []
    for row in examples:
        cs.append(row[-1])
    zupp = list(set(cs))

    csp = []
    for row in parent_examples:
        csp.append(row[-1])
    pzupp = list(set(csp))

    if len(zupp) == 1:
        return zupp[0]
    elif examples.size == 0:
        freq_m = 0
        z_max = 0
        for z in zupp:
            freq = cs.count(z)/len(cs)
            if freq > freq_m:
                z_max = z
        return z_max
    elif len(attributes) == 0:
        freq_m = 0
        z_max = 0
        for z in pzupp:
            freq = csp.count(z)/len(csp)
            if freq > freq_m:
                z_max = z
        return z_max
    else:
        t = Tree()
        ag = list(map(gain,[list([examples])[0]]*len(attributes),attributes))
        m = max(ag)
        a = ag.index(m)
        t.name = attributes[a]
        v = []
        exs = []
        for row in examples:
            v.append(row[attributes[a]])
        supp = list(set(v))
        #print([attributes,a,supp])
        new_att = cp.deepcopy(attributes)
        del new_att[a]
        for e in supp:
            z = [i for i, x in enumerate(v) if x==e]
            exs=ny.array(list(examples[z]))
            subt = dtl(exs,new_att,examples)
            t.sons[e] = subt
        return t


def tree_comp(t,filename,names):
    out_file = open(filename+".dot","w")
    out_file.write("digraph abstract {\n\n")
    tree_exp(t,out_file,names)
    out_file.write("}")
    out_file.close()


def tree_exp(t,outfile,names):
    sons = t.sons
    name = t.name
    for key in sons:
        if type(sons[key]) == Tree:
            outfile.write("    " + str(names[name]) + "->" + str(names[t.sons[key].name]) + " [label=\"" + str(key) + "\"];\n")
            tree_exp(sons[key],outfile,names)
        else:
            outfile.write("    " + str(names[name]) + "->" + str(sons[key]) + " [label=\"" + str(key) + "\"];\n")


def dt_c(t,e):
    a = t.name
    try:
        c = t.sons[e[a]]
    except KeyError:
        return rand.sample([t.sons[i] for i in list(set(t.sons))],1)
    #print(['a,ea,c',a,e[a],c])
    if type(c) != Tree:
        x = c
    else:
        x = dt_c(c,e)
    return x

fil = csv.reader(open('/home/ld/Desktop/irisQ.csv', "rt"))
dataset = ny.array(list(fil))
test_rate = 0.3
#acc_m = []
#for k in range(0,20):
rand_set = rand.sample(range(0,len(dataset)),math.floor(test_rate*len(dataset)))
test = dataset[rand_set]
train = cp.deepcopy(dataset.tolist())
for i in rand_set:
    train[i] = 0
train = ny.array(list(filter(lambda x: x != 0, train)))
#print(train)

tree=dtl(train,list(range(0,len(train[0])-1)),train)
tree_comp(tree,'/home/ld/Desktop/gianfranco',['A0','A1','A2','A3'])
os.system('cd /home/ld/Desktop && dot -Tpng gianfranco.dot -o g.png')
fig = Image.open('/home/ld/Desktop/g.png')
fig.show()
acc = []
for y in test:
    acc.append(dt_c(tree,y)==y[-1])
print(sum(acc)*100/len(acc))
#acc_m.append(sum(acc)*100/len(acc))
#print([round(sum(acc_m)/len(acc_m),2),round(ny.std(acc_m),3),round(ny.var(acc_m)/(sum(acc_m)/len(acc_m)),3)])
