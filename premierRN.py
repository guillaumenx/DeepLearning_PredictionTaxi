# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:02:26 2019

@author: guill
"""

'''
import json
import zipfile
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import pandas as pd
from datetime import datetime
from operator import itemgetter
import imp   

import sys
sys.path.insert(0, 'C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\')
import torch_utils; imp.reload(torch_utils)
#from Loss_Function import Haversine_Loss
from torch_utils import gpu, minibatch, shuffle , Haversine_Loss


data_train=pd.read_csv('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\train.csv',converters={'POLYLINE': lambda x: json.loads(x)})
#data_test=pd.read_csv('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\test.csv',converters={'POLYLINE': lambda x: json.loads(x)})

#j'ai pris un petit nombre juste pour le faire tourner et debuguer
littleTrain=data_train.iloc[:200000][data_train.MISSING_DATA!=True].sample(frac=1)
littleTest=data_train.iloc[-1000:][data_train.MISSING_DATA!=True].sample(frac=1)


#je télécharge le clustering fait dans ClusterGeo
clustering=load('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\clusterDestination.joblib')


#dans arrive y a la destination et chemin c'est le debut de la trajectoire (j'ai prit 10 points)
train_arrive = np.array([ np.asarray(p[-1]) if len(p)>10 else [0,0] for p in littleTrain['POLYLINE']])
train_chemin = np.asarray([ np.asarray(p[:10]).flatten() if len(p)>10 else np.zeros(20) for p in littleTrain['POLYLINE']])
test_arrive = np.array([ np.asarray(p[-1]) if len(p)>10 else [0,0] for p in littleTest['POLYLINE']])
test_chemin = np.asarray([ np.asarray(p[:10]).flatten() if len(p)>10 else np.zeros(20) for p in littleTest['POLYLINE']])


#juste j'enlève les trajectoires non sélectionnées
keepTrain=np.asarray(np.where(train_chemin[:,0]!=0)).reshape(-1)
keepTest=np.asarray(np.where(test_chemin[:,0]!=0)).reshape(-1)
train_arrive=train_arrive[keepTrain]
train_chemin=train_chemin[keepTrain]
test_arrive=test_arrive[keepTest]
test_chemin=test_chemin[keepTest]

#je leur assigne leur label par le cluster entrainé
train_arrive=clustering.predict(train_arrive)

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
for i,cl in enumerate(idClient_train):
    if (int(cl) not in dictClient.keys()):
        dictClient[int(cl)]=i
#je rajoute une dernière case en mettant tous les nan dans cette case
#c-a-d les personnes dont on ne connaît pas l'identité sont considérés comme la même personne
numClNan=int(cl)+1
dictClient[numClNan]=i+1
nbrClient=len(dictClient)

#la il faudrait prendre d'autres métadata mais j'ai pas fait encore
metadata_train=littleTrain[['TAXI_ID','TIMESTAMP','ORIGIN_STAND','DAY_TYPE']].copy()
metadata_test=littleTest[['TAXI_ID','TIMESTAMP','ORIGIN_STAND','DAY_TYPE']].copy()
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
daytyp_train=pd.DataFrame(x[:,3])
daytyp_test=pd.DataFrame(xtest[:,3])
#je vais chercher leur id embeddings dans le dico
daytyp_train=np.asarray([np.asscalar(np.where( jourtype == key)[0]) for key in np.asarray(daytyp_train)])
daytyp_test=np.asarray([np.asscalar(np.where( jourtype == key)[0]) for key in np.asarray(daytyp_test)])

#time
dt_train= [datetime.fromtimestamp(stamp) for stamp in x[:,1]]
week_train=np.asarray([w.date().isocalendar()[1] for w in dt_train])
day_train=np.asarray([d.date().weekday() for d in dt_train])
hour_train=np.asarray([h.hour for h in dt_train])

dt_test = [datetime.fromtimestamp(stamp) for stamp in xtest[:,1]]
week_test=np.asarray([w.date().isocalendar()[1] for w in dt_test])
day_test=np.asarray([d.date().weekday() for d in dt_test])
hour_test=np.asarray([h.hour for h in dt_test])


nbrCluster=1000
'''
#le réseau temporaire
class PredictionDest(nn.Module):
    
    def __init__(self, nbrCluster , nbrTaxi , nbrClient , nbrWeek=52 , nbrH=24 , nbrJ=7 , dim_emb_tx=10 , dim_emb_cl=10 , dim_emb_week=10 , dim_emb_heure=10 , dim_emb_jour=10):
        super(PredictionDest, self).__init__()
        self.embTaxiId=nn.Embedding(num_embeddings=nbrTaxi,embedding_dim=dim_emb_tx)
        self.embClientId=nn.Embedding(num_embeddings=nbrClient,embedding_dim=dim_emb_cl)
        self.embWeek=nn.Embedding(num_embeddings=nbrWeek,embedding_dim=dim_emb_week)
        self.embHour=nn.Embedding(num_embeddings=nbrH,embedding_dim=dim_emb_heure)
        self.embDay=nn.Embedding(num_embeddings=nbrJ,embedding_dim=dim_emb_jour)
        
        
        self.lin1 = nn.Linear(in_features=70,out_features=500)
        self.lin2 = nn.Linear(500,nbrCluster)
        
    def forward(self, path , taxi_ids , client_ids , week , day , hour):
        taxi_emb=self.embTaxiId(taxi_ids)
        client_emb=self.embClientId(client_ids)
        week_emb=self.embWeek(week)
        day_emb=self.embDay(day)
        hour_emb=self.embHour(hour)
        
        x=torch.cat((path , taxi_emb, client_emb , week_emb , day_emb , hour_emb), dim=1)
        
        x = self.lin1(x)
        x=F.relu(x)
        x = self.lin2(x)
        return x
    

def train(model, paths , taxi_ids , client_ids , weeks , days , hours , dest ,loss_fn , optimizer , use_cuda=False ,n_epochs=1,batch_size=32, verbose=True):
    
    model.train(True)
    
    loss_train = np.zeros(n_epochs)
    
    for epoch_num in range(n_epochs):
        paths, taxi_ids, client_ids , weeks , days , hours , destinations = shuffle(paths,taxi_ids,client_ids, weeks , days , hours ,dest)

        paths_tensor = gpu(torch.from_numpy(paths).type(torch.FloatTensor),
                              use_cuda)
        taxi_ids_tensor=gpu(torch.from_numpy(taxi_ids).type(torch.LongTensor),
                              use_cuda)
        client_ids_tensor=gpu(torch.from_numpy(client_ids).type(torch.LongTensor),
                              use_cuda)
        week_tensor=gpu(torch.from_numpy(weeks).type(torch.LongTensor),
                              use_cuda)
        day_tensor=gpu(torch.from_numpy(days).type(torch.LongTensor),
                              use_cuda)
        hour_tensor=gpu(torch.from_numpy(hours).type(torch.LongTensor),
                              use_cuda)
        destinations_tensor = gpu(torch.from_numpy(destinations).type(torch.LongTensor),
                             use_cuda)
        epoch_loss = 0.0

        for (minibatch_num,
             (batch_paths,
              batch_taxis,
              batch_clients,
              batch_week,
              batch_day,
              batch_hour,
              batch_destination)) in enumerate(minibatch(batch_size,
                                                     paths_tensor,
                                                     taxi_ids_tensor,
                                                     client_ids_tensor,
                                                     week_tensor,
                                                     day_tensor,
                                                     hour_tensor,
                                                     destinations_tensor)):
            
            
    
            predictions = model(batch_paths, batch_taxis , batch_clients , batch_week , batch_day , batch_hour)

            optimizer.zero_grad()
            
            loss = loss_fn(predictions , batch_destination)
            
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

def test(model, paths , taxi_ids , client_ids ,  weeks , days , hours , loss_fn, destinations, use_cuda=False):
    model.train(False)

    paths_tensor = gpu(torch.from_numpy(paths).type(torch.FloatTensor),
                              use_cuda)
    taxi_ids_tensor=gpu(torch.from_numpy(taxi_ids).type(torch.LongTensor),
                              use_cuda)
    client_ids_tensor=gpu(torch.from_numpy(client_ids).type(torch.LongTensor),
                              use_cuda)
    week_tensor=gpu(torch.from_numpy(weeks).type(torch.LongTensor),
                          use_cuda)
    day_tensor=gpu(torch.from_numpy(days).type(torch.LongTensor),
                          use_cuda)
    hour_tensor=gpu(torch.from_numpy(hours).type(torch.LongTensor),
                              use_cuda)
    destinations_tensor = gpu(torch.from_numpy(destinations).type(torch.LongTensor),
                             use_cuda)
    predictions = model(paths_tensor, taxi_ids_tensor , client_ids_tensor , week_tensor , day_tensor , hour_tensor)
        
    loss = loss_fn(destinations_tensor, predictions)
    
    return loss.data.item(),predictions
    
taxi_class = PredictionDest(nbrCluster=nbrCluster, nbrTaxi=nbrTaxi, nbrClient=nbrClient)
use_gpu = torch.cuda.is_available()
if use_gpu:
    taxi_class = taxi_class.cuda()

#regarder le fichier torch_utils c'est la distance prise par kaggle
#loss_fn = Haversine_Loss
    
loss_fn= nn.CrossEntropyLoss() 
lr = 1e-3
optimizer_cl = torch.optim.Adam(taxi_class.parameters())
l_t,pred_tr= train(taxi_class, train_chemin , taxi_train_ids , client_train_ids, week_train , day_train , hour_train , train_arrive , loss_fn , optimizer_cl ,use_cuda=True, n_epochs = 3)
