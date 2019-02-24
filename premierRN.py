# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:02:26 2019

@author: guill
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import pandas as pd
import ast
from operator import itemgetter
import imp   
import torch_utils; imp.reload(torch_utils)
#from Loss_Function import Haversine_Loss
from torch_utils import gpu, minibatch, shuffle , Haversine_Loss


data_train=pd.read_csv('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\train.csv')
data_test=pd.read_csv('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\test.csv')

#j'ai pris un petit nombre juste pour le faire tourner et debuguer
littleTrain=data_train.iloc[:10000][data_train.MISSING_DATA!=True].sample(frac=1)
littleTest=data_train.iloc[-1000:][data_train.MISSING_DATA!=True].sample(frac=1)

#dans arrive y a la destination et chemin c'est le debut de la trajectoire (j'ai prit 10 points)
train_arrive=np.zeros(shape=(len(littleTrain),2))
test_arrive=np.zeros(shape=(len(littleTest),2))
train_chemin=[0 for i in range(len(littleTrain))]
test_chemin=[0 for i in range(len(littleTest))]

#je remplis les array et je me fais pas chier si y a pas assez de points je les compte pas pour l'instant
for i in range(len(littleTest)):
    tr=np.asarray(ast.literal_eval(littleTrain['POLYLINE'].iloc[i])).flatten()
    te=np.asarray(ast.literal_eval(littleTest['POLYLINE'].iloc[i])).flatten()
    if(len(tr)>20):
        train_arrive[i]=tr[-2:]
        train_chemin[i]=tr[:20]
    if(len(te)>20):
        test_chemin[i]=te[:20]
        test_arrive[i]=te[-2:]
    
    
for i in range(len(littleTest),len(littleTrain)):
    tr=np.asarray(ast.literal_eval(littleTrain['POLYLINE'].iloc[i])).flatten()
    if(len(tr)>10):
        train_chemin[i]=tr[:20]
        train_arrive[i]=tr[-2:]

#juste j'enlève les trajectoires non sélectionnées
keepTrain=np.asarray(np.where(train_arrive[:,0]!=0)).reshape(-1)
keepTest=np.asarray(np.where(test_arrive[:,0]!=0)).reshape(-1)
train_arrive=train_arrive[keepTrain]
train_chemin=itemgetter(*keepTrain)(train_chemin)
test_arrive=test_arrive[keepTest]
test_chemin=itemgetter(*keepTest)(test_chemin)

#dico taxiid vers embeddingid
#j'ai testé: les taxis présents dans le test set sont présents dans le train
dictTaxi={}
idTaxi_train=np.unique(data_train['TAXI_ID'])
nbrTaxi=len(idTaxi_train)
for i,tx in enumerate(idTaxi_train):
    if(tx not in dictTaxi.keys()):
        dictTaxi[tx]=i

dictClient={}
idClient_train=np.unique(data_train['ORIGIN_CALL'])
idClient_train=idClient_train[~np.isnan(idClient_train)]
nbrClient=len(idClient_train)
for i,cl in enumerate(idClient_train):
    if (int(cl) not in dictClient.keys()):
        dictClient[int(cl)]=i
#je rajoute une dernière case en mettant tous les nan dans cette case
#c-a-d les personnes dont on ne connaît pas l'identité sont considérés comme la même personne
numClNan=int(cl)+1
dictClient[numClNan]=i+1

#la il faudrait prendre d'autres métadata mais j'ai pas fait encore
metadata_train=littleTrain[['TAXI_ID','TIMESTAMP','ORIGIN_STAND']].copy()
metadata_test=littleTest[['TAXI_ID','TIMESTAMP','ORIGIN_STAND']].copy()
#je supprime les données n'ayant pas passé le cut 
x = metadata_train.values[keepTrain] 
xtest=metadata_test.values[keepTest]

taxi_train_ids=x[:,0]
taxi_test_ids=xtest[:,0]
#je vais chercher leur id embeddings dans le dico
taxi_train_ids=np.asarray([dictTaxi.get(key) for key in taxi_train_ids])
taxi_test_ids=np.asarray([dictTaxi.get(key) for key in taxi_test_ids])


client_train_ids=littleTrain['ORIGIN_CALL'].copy()
client_test_ids=littleTest['ORIGIN_CALL'].copy()
client_train_ids[np.isnan(client_train_ids)]=numClNan
client_test_ids[np.isnan(client_test_ids)]=numClNan
client_train_ids=client_train_ids.values[keepTrain]
client_test_ids=client_test_ids.values[keepTest]
#je vais chercher leur id embeddings dans le dico
client_train_ids=np.asarray([dictClient.get(key) for key in client_train_ids])
client_test_ids=np.asarray([dictClient.get(key) for key in client_test_ids])

#il faudrait juste supprimer les lignes comme fait pour les metadonnees précédentes
jourtype=np.array(['A','B','C'])
daytyp_train=littleTrain['DAY_TYPE'].copy()
daytyp_test=littleTest['DAY_TYPE'].copy()
#je vais chercher leur id embeddings dans le dico
daytyp_train=np.asarray([np.asscalar(np.where( jourtype == key)[0]) for key in daytyp_train])
daytyp_test=np.asarray([np.asscalar(np.where( jourtype == key)[0]) for key in daytyp_test])

#origin_train=x[:,2]
#origin_test=xtest[:,2]

#normalisation du temps
time_train=x[:,1].reshape(-1,1)
time_test=xtest[:,1].reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
time_scaled = min_max_scaler.fit_transform(time_train)
timetest_scaled=min_max_scaler.transform(time_test)
#time_train= pd.DataFrame(time_scaled)
#time_test= pd.DataFrame(timetest_scaled)



#le réseau temporaire
class PredictionDest(nn.Module):
    
    def __init__(self, nbrTaxi , dim_emb_tx=10):
        super(PredictionDest, self).__init__()
        self.embTaxiId=nn.Embedding(num_embeddings=nbrTaxi,embedding_dim=dim_emb_tx)
        
        
        self.lin1 = nn.Linear(in_features=31,out_features=1000)
        self.lin2 = nn.Linear(1000,500)
        self.lin3 = nn.Linear(500,2)
        
    def forward(self, path , taxi_ids, time):
        taxi_emb=self.embTaxiId(taxi_ids)
        
        x=torch.cat((taxi_emb,path,time), dim=1)
        
        x = self.lin1(x)
        x=F.relu(x)
        x = self.lin2(x)
        x=F.relu(x)
        x = self.lin3(x)
        return x
    

def train(model, paths , taxi_ids , time , dest ,loss_fn , optimizer , use_cuda=False ,n_epochs=1,batch_size=32, verbose=True):
    
    model.train(True)
    
    loss_train = np.zeros(n_epochs)
    
    for epoch_num in range(n_epochs):
        paths, taxi_ids, time , destinations = shuffle(paths,taxi_ids,time,dest)

        paths_tensor = gpu(torch.from_numpy(paths).type(torch.FloatTensor),
                              use_cuda)
        taxi_ids_tensor=gpu(torch.from_numpy(taxi_ids).type(torch.LongTensor),
                              use_cuda)
        time_tensor=gpu(torch.from_numpy(time).type(torch.FloatTensor),
                              use_cuda)
        destinations_tensor = gpu(torch.from_numpy(destinations).type(torch.FloatTensor),
                             use_cuda)
        epoch_loss = 0.0

        for (minibatch_num,
             (batch_paths,
              batch_taxis,
              batch_time,
              batch_destination)) in enumerate(minibatch(batch_size,
                                                     paths_tensor,
                                                     taxi_ids_tensor,
                                                     time_tensor,
                                                     destinations_tensor)):
            
            
    
            predictions = model(batch_paths, batch_taxis , batch_time)

            optimizer.zero_grad()
            
            loss = loss_fn(batch_destination, predictions)
            
            epoch_loss = epoch_loss + loss.data.item()
            
            loss.backward()
            optimizer.step()
            
        
        epoch_loss = epoch_loss / (minibatch_num + 1)
        loss_train[epoch_num]=epoch_loss

        if verbose:
            print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))
    
        if np.isnan(epoch_loss) or epoch_loss == 0.0:
            raise ValueError('Degenerate epoch loss: {}'
                             .format(epoch_loss))
    
    return loss_train,predictions

def test(model, paths , taxi_ids , time , loss_fn, destinations, use_cuda=False):
    model.train(False)

    paths_tensor = gpu(torch.from_numpy(paths).type(torch.FloatTensor),
                              use_cuda)
    taxi_ids_tensor=gpu(torch.from_numpy(taxi_ids).type(torch.LongTensor),
                              use_cuda)
    time_tensor=gpu(torch.from_numpy(time).type(torch.FloatTensor),
                              use_cuda)
    destinations_tensor = gpu(torch.from_numpy(destinations).type(torch.FloatTensor),
                             use_cuda)
    predictions = model(paths_tensor, taxi_ids_tensor , time_tensor)
        
    loss = loss_fn(destinations_tensor, predictions)
    
    return loss.data.item(),predictions


#mes données d'entrainements
destTr=train_arrive
pathsTr=np.array([np.concatenate((train_chemin[i][:10],train_chemin[i][-10:])) for i in range(len(train_chemin))])

#pathsTr=np.array([np.concatenate((x_scaled[i,:],train_chemin[i][:10],train_chemin[i][-10:])) for i in range(len(x_scaled))])

    
taxi_class = PredictionDest(nbrTaxi)
use_gpu = torch.cuda.is_available()
if use_gpu:
    taxi_class = taxi_class.cuda()

#regarder le fichier torch_utils c'est la distance prise par kaggle
loss_fn = Haversine_Loss
lr = 1e-3
optimizer_cl = torch.optim.Adam(taxi_class.parameters())
l_t,pred_tr= train(taxi_class, pathsTr , taxi_train_ids , time_scaled , destTr, loss_fn , optimizer_cl ,use_cuda=True, n_epochs = 10)


#données de test
pathsTe=np.array([np.concatenate((test_chemin[i][:10],test_chemin[i][-10:])) for i in range(len(test_chemin))])
destTe=test_arrive
