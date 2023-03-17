from re import T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torch.autograd import Variable
import argparse
import time
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")   
import numpy as np
from utils import *
from mydata import *
from models import *
eval_interval = 100
batch_size = 6
decay_iter = 50
decay_lamb = 1

def train(rundir,source_temp,target_temp,source_data_path,source_train_set,source_test_set,target_data_path,target_train_set,target_test_set,models, criterion, optimizers, batch_size, epochs,eval_interval, lamb1,lamb2,lamb3,seed=0, device_type=('cuda:0' if torch.cuda.is_available() else 'cpu'), ifsave=True, load_model=False, model_path='/models/best.pt'):
  loss_min = 10000
  rundir = mkdir(rundir)
  device = torch.device(device_type)
  if load_model:
      load_saved_model(device,models,optimizers,loss_min,seed,model_path)
  if torch.cuda.is_available():
    for  model in models:
      models[model].to(device)
    
  init_seed(seed)
  criterion = criterion
  criterion_mae = nn.L1Loss()
  criterion_mse = nn.MSELoss()
  for temp_idx in range(1):
    source_data = Mydataset(source_data_path, source_temp, source_train_set,mode='train')
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True)
    target_data = Mydataset(target_data_path, target_temp, target_train_set,mode='train')
    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=True)
    target_test_data = Mydataset(target_data_path, target_temp, target_test_set, mode='test')
    target_test_loader = DataLoader(target_test_data, batch_size=1, shuffle=False)
    
    loss_iter_target = []
    loss_iter_A, loss_iter_B, loss_iter_C = [],[],[]

    test_len = len(target_test_loader)
    min_max,min_mae,min_rmse = [],[],[]
    for i in range(test_len):
      min_mae.append(1)
      min_rmse.append(1)
      min_max.append(1)
    #checkpoint = torch.load(load_model_path, map_location=device)
     #models['domain_classifier'].load_state_dict(checkpoint['domain_classifier'])
    for epoch in range(epochs):
      starttime = time.time()
      if ((epoch+1) % decay_iter) == 0:
        lamb3 = lamb3 * decay_lamb
      ##########
      #train
      ##########
      start_step = epoch*len(source_loader)
      for model in models:
        models[model].train()
      loss_train = 0
      loss_A, loss_B, loss_C = 0,0,0
      loss_test = 0
      tqdm_mix = tqdm(zip(source_loader, target_loader),desc='epoch '+str(epoch))
      source_sample = 0
      target_sample = 0
      for i, ((source_data, source_label), (target_data, target_label)) in enumerate(tqdm_mix):
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        target_label = target_label.to(device)
        
        
        #in source-free setting, source label is unaccessible,
        #if source label is accessible, you can remove the comment to run step A and step B
        """
        #step A
        for op in optimizers:
          optimizers[op].zero_grad()
        source_features = models['lstm'](models['conv'](source_data))
        predicts_p1 = models['fc1'](source_features).squeeze()
        predicts_p2 = models['fc2'](source_features).squeeze()
        loss_stepA = lamb1*criterion(predicts_p1,source_label) + criterion(predicts_p2,source_label)
        loss_stepA.backward()
        optimizers['conv'].step()
        optimizers['lstm'].step()
        optimizers['fc1'].step()
        optimizers['fc2'].step()
        
        #step B
        for op in optimizers:
          optimizers[op].zero_grad()
        source_features = models['lstm'](models['conv'](source_data))
        predicts_p1 = models['fc1'](source_features).squeeze()
        predicts_p2 = models['fc2'](source_features).squeeze()
        loss_stepB = lamb1*(criterion(predicts_p1,source_label)\
                   + criterion(predicts_p2,source_label)\
                   - discrepancy(predicts_p1, predicts_p2))
        loss_stepB.backward()
        optimizers['fc1'].step()
        optimizers['fc2'].step() """

        #step C
        for op in optimizers:
          optimizers[op].zero_grad()
        target_features = models['lstm'](models['conv'](target_data))
        predicts_p1 = models['fc1'](target_features).squeeze()
        predicts_p2 = models['fc2'](target_features).squeeze()
        loss_target = criterion(predicts_p1, target_label) + criterion(predicts_p2, target_label)
        loss_stepC = lamb2*discrepancy(predicts_p1, predicts_p2) + lamb3*loss_target
        loss_stepC.backward()
        optimizers['conv'].step()
        optimizers['lstm'].step()
        
        source_sample += len(source_data)
        target_sample += len(target_data)
        loss_train += loss_target.item()
        loss_A += loss_stepA.item()
        loss_B += loss_stepB.item()
        loss_C += loss_stepC.item()
      loss_train = loss_train/(target_sample)
      loss_A = loss_A / source_sample
      loss_B = loss_B / source_sample
      loss_C = loss_C / target_sample
      
      endtime = time.time()
      print('time:',endtime-starttime)
      print('epoch {}:loss {} loss_A {} loss_B {} loss_C {}'.format(epoch, loss_train, loss_A, loss_B, loss_C))

      if ( ((epoch+1) % eval_interval)==0 ) & (ifsave==True):
        save_model(models, optimizers, loss_min, seed, model_path='./saved_model/epoch'+str(epoch)+'.pt')
    plot_train_loss(rundir,loss_iter_target, loss_iter_A, loss_iter_B, loss_iter_C, epochs)