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
data_test=pd.read_csv('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\test.csv',converters={'POLYLINE': lambda x: json.loads(x)})


#j'ai pris un petit nombre juste pour le faire tourner et debuguer
data_train = data_train[data_train['POLYLINE'] != '[]']
data_test = data_test[data_test['POLYLINE'] != '[]']
littleTrain=data_train[data_train.MISSING_DATA!=True].sample(frac=1)
littleTest=data_test[data_test.MISSING_DATA!=True].sample(frac=1)



littleTrain = pd.read_pickle('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\littleTrain.pkl')
littleTest = pd.read_pickle('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\data\\littleTest.pkl')


littleTrain['DURATION'] = littleTrain['POLYLINE'].apply(lambda x: 15 * len(x))
littleTest['DURATION'] = littleTest['POLYLINE'].apply(lambda x: 15 * len(x))

indices = np.where((littleTrain.DURATION > 150) & (littleTrain.DURATION <= 2 * 3600))
littleTrain = littleTrain.iloc[indices]
indices = np.where((littleTest.DURATION > 150) & (littleTest.DURATION <= 2 * 3600))
littleTest = littleTest.iloc[indices]


#je télécharge le clustering fait dans ClusterGeo
clustering=load('C:\\Users\\guill\\Documents\\Cours\\Polytechnique\\3A\\MAP583 Apprentisage Profond\\Projet\\clusterDestinationMS.joblib')

#dans arrive y a la destination et chemin c'est le debut de la trajectoire (j'ai prit 10 points)
train_arrive = np.array([ np.asarray(p[-1]) for p in littleTrain['POLYLINE']])
train_chemin = np.asarray([ np.asarray(p[:-1]).flatten() for p in littleTrain['POLYLINE']])
test_arrive = np.array([ np.asarray(p[-1]) for p in littleTest['POLYLINE']])
test_chemin = np.asarray([ np.asarray(p[:-1]).flatten() for p in littleTest['POLYLINE']])

# Remove trips that are too far away from Porto (also likely due to GPS issues)
bounds = (  # Bounds retrieved using http://boundingbox.klokantech.com
    (-8.727951, 41.052431),
    (-8.456039 , 41.257678)
)
keepTrain = np.where(
    (train_arrive[:,0]  >= bounds[0][0]) &
    (train_arrive[:,1] >= bounds[0][1]) &
    (train_arrive[:,0]  <= bounds[1][0]) &
    (train_arrive[:,1] <= bounds[1][1])
)

train_arrive = train_arrive[keepTrain[0]]
train_chemin = train_chemin[keepTrain[0]]


train_chemin=np.asarray([ train_chemin[i][:np.random.randint(20,len(train_chemin[i])+1)] for i in range(len(train_chemin))])

train_chemin=np.asarray([ np.concatenate((train_chemin[i][:10],train_chemin[i][-10:])) for i in range(len(train_chemin))])
test_chemin=np.asarray([ np.concatenate((test_chemin[i][:10],test_chemin[i][-10:])) for i in range(len(test_chemin))])


scaler = preprocessing.StandardScaler()
scaler2 = preprocessing.StandardScaler()
train_chemin[:,::2]= scaler.fit_transform(train_chemin[:,::2].flatten().reshape(-1,1)).reshape(train_chemin.shape[0],-1)
train_chemin[:,1::2]= scaler2.fit_transform(train_chemin[:,1::2].flatten().reshape(-1,1)).reshape(train_chemin.shape[0],-1)
test_chemin[:,::2]=scaler.transform(test_chemin[:,::2].flatten().reshape(-1,1)).reshape(test_chemin.shape[0],-1)
test_chemin[:,1::2]=scaler2.transform(test_chemin[:,1::2].flatten().reshape(-1,1)).reshape(test_chemin.shape[0],-1)


#dico taxiid vers embeddingid
#j'ai testé: les taxis présents dans le test set sont présents dans le train

leTx = preprocessing.LabelEncoder()
leCl = preprocessing.LabelEncoder()

#la il faudrait prendre d'autres métadata mais j'ai pas fait encore
metadata_train=littleTrain[['TAXI_ID','TIMESTAMP','ORIGIN_STAND']].copy()
metadata_test=littleTest[['TAXI_ID','TIMESTAMP','ORIGIN_STAND']].copy()
#je supprime les données n'ayant pas passé le cut 
x = metadata_train.values[keepTrain] 
#xtest=metadata_test.values[keepTest]
xtest=metadata_test.values

taxi_train_ids=x[:,0]
nbrTaxi=len(np.unique(taxi_train_ids))
taxi_test_ids=xtest[:,0]

leTx.fit(taxi_train_ids)
taxi_train_ids=leTx.transform(taxi_train_ids)
taxi_test_ids=leTx.transform(taxi_test_ids)


client_train_ids=littleTrain['ORIGIN_CALL'].copy()
client_test_ids=littleTest['ORIGIN_CALL'].copy()
client_train_ids[np.isnan(client_train_ids)]=0
client_test_ids[np.isnan(client_test_ids)]=0
client_train_ids=client_train_ids.values[keepTrain]
client_test_ids=client_test_ids.values
nbrClient=len(np.unique(client_train_ids))

leCl.fit(np.concatenate((client_train_ids,client_test_ids)))
client_train_ids=leCl.transform(client_train_ids)
client_test_ids=leCl.transform(client_test_ids)

#time
dt_train= [datetime.fromtimestamp(stamp) for stamp in x[:,1]]
week_train=np.asarray([w.date().isocalendar()[1] for w in dt_train])
day_train=np.asarray([d.date().weekday() for d in dt_train])
hour_train=np.asarray([h.hour*4+int(h.minute/15) for h in dt_train])

dt_test = [datetime.fromtimestamp(stamp) for stamp in xtest[:,1]]
week_test=np.asarray([w.date().isocalendar()[1] for w in dt_test])
day_test=np.asarray([d.date().weekday() for d in dt_test])
hour_test=np.asarray([h.hour*4+int(h.minute/15) for h in dt_test])

nbrCluster=1000
'''
#le réseau temporaire
class PredictionDest(nn.Module):
    
    def __init__(self, nbrCluster , nbrTaxi , nbrClient , nbrWeek=52 , nbrH=96 , nbrJ=7 , dim_emb_tx=10 , dim_emb_cl=10 , dim_emb_week=10 , dim_emb_heure=10 , dim_emb_jour=10):
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
        x= F.softmax(x)
        return x


taxi_class = PredictionDest(nbrCluster=nbrCluster, nbrTaxi=nbrTaxi, nbrClient=nbrClient)
use_gpu = torch.cuda.is_available()
if use_gpu:
    taxi_class = taxi_class.cuda()


#regarder le fichier torch_utils c'est la distance prise par kaggle
loss_fn = Haversine_Loss
    
#loss_fn= nn.CrossEntropyLoss() 
lr = 1e-3
optimizer_cl = torch.optim.SGD(taxi_class.parameters(),lr=lr,momentum=0.9)


def train(model, paths , taxi_ids , client_ids , weeks , days , hours , dest , cluster_centers , loss_fn , optimizer , use_cuda=False ,n_epochs=1,batch_size=32, verbose=True):
    
    model.train(True)
    
    loss_train = np.zeros(n_epochs)
    
    centers= gpu(torch.from_numpy(cluster_centers).type(torch.FloatTensor),
                              use_cuda)
    
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
        destinations_tensor = gpu(torch.from_numpy(destinations).type(torch.FloatTensor),
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

            predictions = torch.mm(predictions,centers)

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

def test(model, paths , taxi_ids , client_ids ,  weeks , days , hours , loss_fn, destinations, cluster_centers , use_cuda=False):
    model.train(False)

    centers= gpu(torch.from_numpy(cluster_centers).type(torch.FloatTensor),
                              use_cuda)
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
    
    predictions = torch.mm(predictions,centers)
    
    loss = loss_fn(predictions, destinations_tensor)
    
    return loss.data.item(),predictions
    

l_t,pred_tr= train(taxi_class, train_chemin , taxi_train_ids , client_train_ids, week_train , day_train , hour_train , train_arrive , clustering.cluster_centers_, loss_fn , optimizer_cl ,use_cuda=True, n_epochs = 2)

_,predLab=torch.max(pred_tr,1)
#running_corrects+=(predLab==batch_theft).sum()
