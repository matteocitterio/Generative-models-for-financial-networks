import numpy as np
import os
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
import dataset_managment
import model_managment
import train_managment
import torch_geometric

from GCLSTM import GCLSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Fix current device
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print('Current device:', device)

#This lines read and manage model parameters by reading the `params.yaml` file. For more info, check `Sandbox_utils.py` 
parser = utils.create_parser()
args = utils.parse_args(parser)

print('\n',args, '\n')

# Retrieve dataset from the simulation
path_name = '../Definitive/data/'
data_file_name = path_name + f'subgraphs_Duffie_{args.num_nodes}nodes_3gamma.pt'

try:
    print(f'Retrieving data...')
    dataset = torch.load(path_name + f'subgraphs_Duffie_{args.num_nodes}nodes_3gamma.pt',  map_location=torch.device(device))

except:
    print('Error: the data file doesnt exist, please run `SimulateNetwork.py`')
    raise ValueError

# Generate CIR process & simulation (The same used for simulating the network)
sim = Simulation(args.alpha, args.b, args.sigma, args.v_0, args.years, gamma = args.gamma, seed=True)
CIRProcess = torch.tensor(sim.CIRProcess.reshape(-1,1)).to(torch.float32).to(device)

# Train-test split
training_index = int(0.8 * len(dataset))

print('Dataset len: ', len(dataset))

#TRAIN
train_dataset = dataset[:training_index]

#TEST
test_dataset = dataset[training_index:]
       
# Slice dataset into windows
Contract_train, y_margin_train, r_train = utils.create_graph_windows(args, device, train_dataset)
Contract_test,y_margin_test, r_test = utils.create_graph_windows(args, device, test_dataset)

#Dataset for the gc-lstm

########################
n_samples = len(train_dataset) - args.lookback - args.steps_ahead + 1
loop = tqdm(range(n_samples), desc='n_samples')
print('n_samples: ',n_samples)

#######################

gclstm_train_dataset = dataset_managment.DataSplit(args, train_dataset)
gclstm_test_dataset = dataset_managment.DataSplit(args, test_dataset)
torch.manual_seed(0)
gclstm_train_loader = torch_geometric.loader.DataLoader(gclstm_train_dataset, batch_size=args.batch_size, shuffle=True)
gclstm_test_loader = torch_geometric.loader.DataLoader(gclstm_test_dataset, batch_size=args.batch_size, shuffle=True)


train_dataset = TensorDataset(Contract_train.to(device), y_margin_train.to(device), r_train.to(device))
test_dataset = TensorDataset(Contract_test.to(device), y_margin_test.to(device), r_test.to(device))

#Transform it into a dataloader
torch.manual_seed(0)
loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
loader_test = DataLoader(test_dataset, shuffle=True, batch_size = args.batch_size)

#Let's get towards the training:
criterion = nn.MSELoss()           #Loss function

#Define the model
mymodel = model_managment.CompleteModel(args).to(device)   
print(mymodel)

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
    train_loss = utils.do_epoch(args, mymodel, loader, gclstm_train_loader, criterion, optimizer)

    #Scheduler step
    scheduler.step()

    #Track the average over the batches
    training_loss.append(train_loss)
 
    #Validation every 50 epochs
    if epoch % args.validation_every ==0:

        with torch.no_grad():
            #Set the model in eval mod
            mymodel.eval()

            test_loss = utils.do_epoch(args, mymodel, loader_test, gclstm_test_loader, criterion, optimizer, training=False)

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
print('Performing prediction')
train_margin_labels, train_benchmark, train_margin_predictions = utils.predictions(args, train_dataset, mymodel)
test_margin_labels, test_benchmark, test_margin_predictions = utils.predictions(args, test_dataset, mymodel)

np.save(f'./results/train_label_{args.steps_ahead}.npy',train_margin_labels)
np.save(f'./results/train_pred_{args.steps_ahead}.npy',train_margin_predictions)
np.save(f'./results/train_benchmark_{args.steps_ahead}.npy', train_benchmark)
np.save(f'./results/test_label_{args.steps_ahead}.npy',test_margin_labels)
np.save(f'./results/test_pred_{args.steps_ahead}.npy',test_margin_predictions)
np.save(f'./results/test_benchmark_{args.steps_ahead}.npy', test_benchmark)

