# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:36:14 2019

@author: guill
"""

from sklearn.cluster import KMeans,MeanShift
import numpy as np
import ast
import time
import matplotlib.pyplot as plt
import pandas as pd

#train=data_train[data_train.MISSING_DATA!=True].sample(frac=1)
data_train=pd.read_csv('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\train.csv')
train=data_train.iloc[:200000][data_train.MISSING_DATA!=True].sample(frac=1)

#dans arrive y a la destination et chemin c'est le debut de la trajectoire (j'ai prit 10 points)
train_arrive=np.zeros(shape=(len(train),2))

#je remplis les array et je me fais pas chier si y a pas assez de points je les compte pas pour l'instant
for i in range(len(train)):
    tr=np.asarray(ast.literal_eval(train['POLYLINE'].iloc[i])).flatten()
    if(len(tr)>2):
        train_arrive[i]=tr[-2:]

deleteTrain=np.asarray(np.where(train_arrive[:,0]!=0)).reshape(-1)
train_arrive=train_arrive[deleteTrain]

print("debut")
'''
start=time.time()
clustering = KMeans(n_clusters=1000).fit(train_arrive)
end=time.time()
print(end-start)
'''
start=time.time()
clustering = MeanShift(bandwidth=1000).fit(train_arrive)
end=time.time()
print(end-start)


from joblib import dump, load
dump(clustering, 'clusterDestinationMS.joblib') 
