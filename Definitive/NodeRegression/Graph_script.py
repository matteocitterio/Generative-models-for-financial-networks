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

parser = utils.create_parser()
args = utils.parse_args(parser)

path_name = '/u/mcitterio/temp/Generative-models-for-financial-networks/NodeRegression/'
data_file_name = path_name + f'subgraphs_Duffie_{args.num_nodes}nodes_3gamma.pt'

try:
    print(f'Retrieving data...')
    dataset = torch.load(path_name + f'subgraphs_Duffie_{args.num_nodes}nodes_3gamma.pt',  map_location=torch.device(device))

except:
    print('Error: the data file doesnt exist, please run `SimulateNetwork.py`')
    raise ValueError

# Generate CIR process & simulation
sim = Simulation(args.alpha, args.b, args.sigma, args.v_0, args.years, gamma = args.gamma, seed=True)
CIRProcess = torch.tensor(sim.CIRProcess.reshape(-1,1)).to(torch.float32).to(device)

# Define the splitter class which will handle data windowing

# ##################################################
# NumHistSteps = args.lookback
# TrainSplit = args.train_split

# #Used for extrampolation, if == 1, no extrapolation will be performed
# number_of_predictions = args.steps_ahead

# train_dataset = dataset[:TrainSplit]


# # Total number of Datapoints that is possible to make
# total_len = len(dataset) - 2 * NumHistSteps - 2 * (number_of_predictions-1)

#         # Set start and end training indices
# start = NumHistSteps
# end = int(np.floor(total_len*TrainSplit) + NumHistSteps)

#         # Create training split
        
# TrainSet = DataSplit(args, dataset, start, end)
# TrainSet = DataLoader(TrainSet, batch_size=args.batch_size, shuffle=True)


# #################################################


splitter = dataset_managment.Splitter(args, dataset, val_batch_size = args.batch_size)

torch.save(splitter.train, 'train_dataset.pth')

#Let's get towards the training:
criterion = nn.MSELoss()           #Loss function

#Define the model
n_contract_features = 6
mymodel = model_managment.GC_LSTM_model(args).to(device)    
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

loop = tqdm(range(args.n_epochs))

for epoch in loop:

    #Set model in training mode
    mymodel.train()

    temp_loss = 0

    for s in splitter.train:

        # print(s.keys())
        # print('len(hist_...)', len(s['hist_adj_list']))
        # print(s['hist_adj_list'][0])
        # print(s['hist_adj_list'][0].x)
        # print(s['hist_adj_list'][0].edge_index)
        # print(s['hist_adj_list'][0].batch)

        loss = train_managment.run_epoch(args, mymodel, s, criterion)
        
        #Backpropagation
        loss.backward()

        #Optimizer step
        optimizer.step() 
        optimizer.zero_grad()

        #Track the batch loss
        temp_loss += loss.item()
        
        # loop.set_postfix(loss=loss.item())
    
    train_loss = temp_loss/len(splitter.train)
    #loop.set_postfix(loss = train_loss)

    #Scheduler step
    scheduler.step()

    #Track the average over the batches
    training_loss.append(train_loss)
 
    #Validation every 50 epochs
    if epoch % args.validation_every ==0:

        with torch.no_grad():
            #Set the model in eval mod
            mymodel.eval()
            test_loss = 0
            for s in splitter.val:
                loss = train_managment.run_epoch(args, mymodel, s, criterion)
                test_loss+=loss.item()

            #Track the average loss over the batches
            validation_loss.append(test_loss/len(splitter.val))

        #Check early stopping
        early_stopping(validation_loss[-1], mymodel)
        if early_stopping.early_stop:

            #Load the best set of parameters
            mymodel.load_state_dict(torch.load(f'temp_state_dict{mymodel.name}.pt'))
            print(f"Early stopping at epoch {epoch}.")
            break

    lrs.append(optimizer.param_groups[0]["lr"])

    #print('es_:', early_stopping)
            
    #Give informations in the loop
    loop.set_postfix(loss = train_loss, val_loss = validation_loss[-1], best_val_loss = early_stopping.best_score, counter=early_stopping.counter, lr= lrs[-1])

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

