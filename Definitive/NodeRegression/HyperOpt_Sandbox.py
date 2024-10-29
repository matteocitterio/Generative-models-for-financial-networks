import numpy as np
from CIR import get_CIR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from simulation import Simulation, Contract
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
import Sandbox_utils as utils
import HyperOpt_utils as hyper_utils

#For the hyperopt using ray
from functools import partial
import os
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print('Current device:', device)


parser = utils.create_parser()
args = utils.parse_args(parser)

loader, loader_test = hyper_utils.DatasetBuilding(args)

#define the model
class model(nn.Module):

    def __init__(self, lstm_size, l1, number_layers, l2):
        super(model, self).__init__()

        self.input_size = 5#n_contract_features
        self.name = l1# args.regressor_hidden_size

        self.lstm = torch.nn.LSTM(input_size = self.input_size, 
                                  hidden_size = lstm_size,#args.lstm_hidden_size, 
                                  num_layers = 1)
        
        # Initialize the regressor layers
        layers = [nn.Linear(lstm_size + 1, l1),
                  nn.ReLU()]
        
        #Add an additional layer
        for _ in range(number_layers - 1):

            layers.extend([nn.Linear(l1, l2),
                           nn.ReLU()])
            
        # Add the output layer
        layers.append(nn.Linear(l2 if number_layers > 1 else l1, 1))

        # Create the regressor using nn.Sequential
        self.regressor = nn.Sequential(*layers)
    
        #Reset parameters of the model
        self.reset_parameters()

    def forward(self, x, r):
        # Flatten LSTM parameters
        self.lstm.flatten_parameters()

        pred = torch.zeros((x.shape[0], 1)).to(torch.float32).to(device)

        for i in range(x.shape[2]//self.input_size):
            
            contract = x[:,:, i*self.input_size : (i+1)*self.input_size] 
            #if the contract is non empty:
            if not torch.all(torch.eq(contract, torch.zeros_like(contract))):
                lstm_hidden, _ = self.lstm(contract)
                pred+=self.regressor(torch.squeeze(torch.cat([lstm_hidden[:,-1,:], r],dim=1)))
         
        return pred
    
    def reset_parameters(self):

        self.lstm.reset_parameters()
        for layer in [self.regressor]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                
    
#Let's get towards the training:
criterion = nn.MSELoss()           #Loss function

#Define the model
mymodel = model().to(device)    

#optimizer & scheduler
optimizer = torch.optim.Adam(mymodel.parameters(), lr = args.initial_lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.scheduler_milestone_frequency * i for i in range(args.scheduler_number_of_updates)], gamma=args.scheduler_gamma)

#Define an early stopping object
early_stopping = utils.EarlyStopping(args.patience)

#Define two lists to store training record
training_loss = []
validation_loss = []
lrs = []

#Define a tqdm loop to get a nice progress bar
loop = tqdm(range(args.n_epochs))

for epoch in loop:

    #Set model in training mode
    mymodel.train()

    #Perform an epoch of training
    train_loss = utils.do_epoch(mymodel, loader, criterion, optimizer)

    #Scheduler step
    scheduler.step()

    #Track the average over the batches
    training_loss.append(train_loss)
 
    #Validation every 50 epochs
    if epoch % args.validation_every ==0:

        with torch.no_grad():
            #Set the model in eval mod
            mymodel.eval()

            test_loss = utils.do_epoch(mymodel, loader_test, criterion, optimizer, training=False)

            #Track the average loss over the batches
            validation_loss.append(test_loss)

        #Check early stopping
        early_stopping(validation_loss[-1], mymodel)
        if early_stopping.early_stop:

            #Load the best set of parameters
            mymodel.load_state_dict(torch.load(f'temp_state_dict{mymodel.name}.pt'))
            print(f"Early stopping at epoch {epoch}.")
            break

    lrs.append(optimizer.param_groups[0]["lr"])
            
    #Give informations in the loop
    loop.set_postfix(loss = train_loss, val_loss = test_loss, best_val_loss = early_stopping.best_score, counter=early_stopping.counter, lr= lrs[-1])

# #Get the predictions
# train_margin_labels, train_margin_predictions = utils.predictions(train_dataset, mymodel)
# test_margin_labels, test_margin_predictions = utils.predictions(test_dataset, mymodel)


mymodel = model(config['lstm_size'], config['l1'], config['number_layers'], config['l2'])

checkpoint = session.get_checkpoint()

if checkpoint:
    checkpoint_state = checkpoint.to_dict()
    start_epoch = checkpoint_state["epoch"]
    net.load_state_dict(checkpoint_state["net_state_dict"])
    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
else:
    start_epoch = 0

