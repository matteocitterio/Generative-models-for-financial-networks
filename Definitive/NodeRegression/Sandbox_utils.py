import numpy as np
import argparse
import sys
import yaml
import os
import re
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def create_parser():
    """
    Selects the parameter file from a command line input. The DEAFULT NAME is `params.yaml`
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file',default='params.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    return parser

def parse_args(parser, save=True):
    """
    Takes a parser in input which tells which file we'll be working on and returns an "args" object.
    RETURN
    args
    """
    args = parser.parse_args()
    if args.config_file:
        data = yaml.safe_load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__

        # Get the args from the params.yaml file
        for key, value in data.items():
            arg_dict[key] = value

        # Get the test names
        path_folder = './parameters/'
        try:
            file_names = [f for f in os.listdir(path_folder) if os.path.isfile(os.path.join(path_folder, f))]
            matches = []
            for file in file_names:
                try: 
                    match = re.search(r'\d+', file).group() 
                    matches.append(int(match))
                except: pass

            matches.sort()
            new_test_number = matches[-1]+1

        except: 
            # First file in the folder
            new_test_number = 1

        new_params_filename = "./parameters/test_"+str(new_test_number)+".yaml"

        # Check if the new file already exists
        if os.path.exists(new_params_filename):
            print(f"Error: File '{new_params_filename}' already exists. Updating test_name")
            sys.exit(1)  # Exit the program with a non-zero exit code

        # Make the validation name coherent to the training name
        args.output_validation_file_name = './results/metrics_test_'+str(new_test_number)+'_valid.txt'
        args.output_training_file_name = './results/metrics_test_'+str(new_test_number)+'.txt'

        args.test_number = new_test_number

        if save:

            # Save the data to a new YAML file with the new filename
            with open(new_params_filename, 'w') as new_file:
                yaml.dump(args, new_file, default_flow_style=False)

    return args

#MAYBE TO DELETE
def CreateContracts(arrival_times, sim):
    """
    Build the contracts using Simulation and arrival times.
    Parameters
    ----------
    - arrival_times : `list`
        List containing the contract's arrival times
    - sim : `Simulation`
        Object of class Simulation
    
    Returns
    -----------
    - `list` : list of class `Contract` objects with the proper starting date
    """
    
    contracts = []
    for arrival_time in arrival_times:
        contract = Contract(arrival_time, sim)
        contracts.append(contract)

    return contracts

#MAYBE TO DELETE, THERE IS ALREADY A CLASS SIM METHOD FOR THIS
def GetMaximumNActiveContracts(contracts, sim):
    """
    Compute maximum number of simultaneously active contracts

    Parameters
    ----------
    - contracts : `list`
        List of objects of class `Contracts`
    - sim : `Simulation`
        Object of class Simulation

    Returns
    ---------
    - max_n_active_contracts : `int`
        The maximum number of active contracts that has been observed over the simulation horizon period
    - days_with_active_contracts : `int`
        Number of days within the simulation horizon with at least an active contract
    """

    max_n_active_contracts = 0
    days_with_active_contracts = 0
    for t in range(sim.TotPoints):
        n_active_contracts = np.sum([contract.is_active(t) for contract in contracts])
        if n_active_contracts > max_n_active_contracts:
            max_n_active_contracts = n_active_contracts
        if n_active_contracts > 0:
            days_with_active_contracts +=1
    
    return max_n_active_contracts, days_with_active_contracts

def create_windows(args, device, features, targets_margin, targets_benchmark, r):
    """
    Windows the dataset into sliding windows of training

    Parameters
    ----------
    - args : `argparser`
        args of the script
    - device : `string`
        used to infer the used device
    - features : `torch.Tensor`
        X features of our problem
    - targets_margin : `torch.Tensor`
        y of our problem
    - targets_benchmark : `torch.Tensor`
        expected value for M_t_l+m
    - r : `torch.Tensor`
        conditioning tensor for our prediction (in this case the interest rate)

    Returns
    ----------
    - X_data : `torch.Tensor`
        windowed X_data
    - y_margin_data : `torch.Tensor`
        windowed y_margin_data
    - r_data : `torch.Tensor`
        windowed r_data
    - y_benchmark_data : `torch.Tensor`
        windowed benchmark (expected M_t_l+m) data
    """
    n_samples = features.shape[0] - args.lookback - args.steps_ahead + 1

    X_data = torch.zeros(n_samples, args.lookback, features.shape[1])
    y_margin_data = torch.zeros(n_samples, args.steps_ahead)
    y_benchmark_data = torch.zeros(n_samples, args.steps_ahead)
    r_data = torch.zeros(n_samples, args.steps_ahead)

    loop = tqdm(range(n_samples), desc='n_samples')

    for i_sample in loop:

        X_data[i_sample, :] = features[i_sample : i_sample + args.lookback]   #it takes data in [i_sample, i_sample+lookback)
        y_margin_data[i_sample] = targets_margin[i_sample + args.lookback : i_sample + args.lookback + args.steps_ahead]
        y_benchmark_data[i_sample] = targets_benchmark[i_sample + args.lookback]
        r_data[i_sample] = r[i_sample + args.lookback : i_sample + args.lookback + args.steps_ahead] - r[i_sample + args.lookback -1 : i_sample + args.lookback + args.steps_ahead -1] #differenziale sa Dio perchè
        
    return X_data.to(torch.float32).to(device), y_margin_data.to(torch.float32).to(device), r_data.to(torch.float32).to(device),  y_benchmark_data.to(torch.float32).to(device)

def create_graph_windows(args, device, y_bench, dataset):
    """
    Windows the dataset into sliding windows of training

    Parameters
    ----------
    - args : `argparser`
        args of the script
    - device : `string`
        used to infer the used device
    - dataset : `List(torch.geometric.Data)`
        List of graphs representing the dataset

    Returns
    ----------
    - X_data : `torch.Tensor`
        windowed X_data
    - y_margin_data : `torch.Tensor`
        windowed y_margin_data
    - edge_index_data: `list`
        windowed edge_indexis tensors
    - r_data : `torch.Tensor`
        windowed r_data
    - y_benchmark_data : `torch.Tensor`
        windowed benchmark (expected M_t_l+m) data
    
    """

    #FOR THE MOMENT WE NEGLECT Y_BENCHMARK

    #Features shape
    features_shape = dataset[0].x.shape[1]

    #Number of windows that we can create
    n_samples = len(dataset) - args.lookback - args.steps_ahead + 1

    #Get the maximum shape of the edge_indexis across the dataset
    #max_edge_shape = max(dataset[i].edge_index.shape[1] for i in range(len(dataset)))

    X_data = torch.zeros(n_samples, args.lookback, args.num_nodes, features_shape)
    y_margin_data = torch.zeros(n_samples, args.num_nodes, args.steps_ahead)
    y_benchmark_data = torch.zeros(n_samples, args.num_nodes, args.steps_ahead)
    r_data = torch.zeros(n_samples, args.steps_ahead)

    loop = tqdm(range(n_samples), desc='n_samples')

    for i_sample in loop:

        X_data[i_sample, :] = torch.stack([dataset[j].x for j in range(i_sample, i_sample + args.lookback)])   # it takes data in [i_sample, i_sample+lookback)
        y_margin_data[i_sample] = torch.hstack([dataset[j].y.reshape(-1,1) for j in range(i_sample + args.lookback, i_sample + args.lookback + args.steps_ahead)])
        # print(y_bench[i_sample: i_sample + args.lookback].shape)
        # print(y_benchmark_data[i_sample].shape)
        # print(y_margin_data[i_sample].shape)
        y_benchmark_data[i_sample] = y_bench[i_sample+ args.lookback : i_sample + args.lookback +args.steps_ahead,:,-1]
        r_data[i_sample] = torch.hstack([dataset[j].r -dataset[j-1].r for j in range(i_sample + args.lookback, i_sample + args.lookback + args.steps_ahead ) ])  #differenziale sa Dio perchè
        
    return X_data.to(torch.float32).to(device), y_margin_data.to(torch.float32).to(device), r_data.to(torch.float32).to(device),  y_benchmark_data.to(torch.float32).to(device)

def remove_edge_index_padding(edge_index):
    """
    Removes the padding from an edge_index tensor.

    Parameters
    ----------
    - edge_index: `torch.tensor`
        padded edge_index tensor with batch_size as first dimension
    Returns
    ----------
    - edge_index: `torch.tensor`
        edge_index with removed padding
    """

    # Find the index of the last non-zero element in each row
    non_zero_indices = (edge_index != 0).sum(dim=1)

    # Find the maximum index of non-zero elements in each row
    max_non_zero_index = non_zero_indices.max()

    # Slice the tensor up to the index of the last non-zero element
    return edge_index[:, :max_non_zero_index + 1]

class EarlyStopping:
    """
    Class that implements EarlyStopping callback on our models
    """
    
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = 10
        self.early_stop = False

    def __call__(self, val_loss, model):

        # print('inside callback: val loss-> ',val_loss)
        # print('best score: -> ',self.best_score)
        # print('counter before: ->',self.counter)
    
        if val_loss > self.best_score:

            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:

            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        
        # print('counter after: ->', self.counter)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), f'temp_state_dict{model.name}.pt')


#Makes the predictions
def graph_predictions(args, model, loader, gc_loader):
    """
    Performs the model evalutation over the pytorch dataset

    Parameters
    ----------
    - dataset : `pytorch.Dataset`
        Dataset over which we want to base our prediction
    - model : `pytorch.nn.Module`
        model we want to use for the prediction
    """

    #Define a dataloader from the dataset without 


    #Define two tensors
    y_margin_preds = torch.zeros(len(loader), args.steps_ahead)
    y_margin_trues = torch.zeros(len(loader), args.steps_ahead)
    #y_benchmark_trues = torch.zeros(len(dataloader), args.steps_filename)

    loop = tqdm(zip(gc_loader,loader), desc='Prediction')
    
    for i_sample, batch_gc, (Contract_batch, y_margin_batch, r_batch) in enumerate(loop):
        with torch.no_grad():

            intensity_embedding = model.gclstm(batch_gc)


            pred = model.lstm(Contract_batch, intensity_embedding.reshape(Contract_batch.shape[0], Contract_batch.shape[2], intensity_embedding.shape[1]), r_batch )
            
            y_preds = model.forward_pred(X,r).cpu()
            # FOCUS ON FIXED LEG
            y_margin_preds[i_sample] = y_preds
            y_margin_trues[i_sample]  = y_margin.cpu()
            y_benchmark_trues[i_sample]  = y_benchmark.cpu()
            
    return y_margin_trues.numpy(), y_benchmark_trues.numpy(), y_margin_preds.numpy()


def do_epoch(args, model, loader, gc_loader, criterion, optimizer, training=True):
    """
    Performs an epoch (both for validation and training) out our model
    """

    temp_loss = 0
    # torch.autograd.set_detect_anomaly(True)

    hihi = 0
    
    for batch_gc, (Contract_batch, y_margin_batch, r_batch) in zip(gc_loader, loader):
        
        #First we operate with the GC_LSTM model:
        # print(hihi, '/',len(gc_loader))
        # hihi+=1

        intensity_embedding = model.gclstm(batch_gc)

        # print(intensity_embedding.reshape(Contract_batch.shape[0], Contract_batch.shape[2], intensity_embedding.shape[1]).shape)

        pred = model.lstm(Contract_batch, intensity_embedding.reshape(Contract_batch.shape[0], Contract_batch.shape[2], intensity_embedding.shape[1]), r_batch )
        # print('pred.shape: ',pred.shape)
        # print('y_margin_batch.shape: ',y_margin_batch.shape)

        loss = criterion(pred, y_margin_batch)

        # print('loss: ',loss)

        #If we are actually training the model instead of performing validation
        if training:

            #Backpropagation
            loss.backward()

            #Optimizer step
            optimizer.step() 
            optimizer.zero_grad()

        #Track the batch loss
        temp_loss += loss.item()

    #Take the average loss over the batches
    return temp_loss / len(loader)


    #     raise NotImplementedError

    # #Instantiate a temp variable that tracks the average loss for the batches
    # temp_loss = 0
    

    # #Loop over data batches
    # for X_batch, y_margin_batch, r_batch, edge_index_batch in loader:

    #     #.forward() method of the model, will output both V_float and V_fixed
    #     print('X_batch_shape: ',X_batch.shape)
    #     print('y_batch_shape: ',y_margin_batch.shape)
    #     print('r_batch_shape: ',r_batch.shape)
    #     print('edge_index_batch shape: ', edge_index_batch.shape)

    #     raise NotImplementedError

    #     y_pred = model(X_batch, edge_index_batch, r_batch)
    #     # print('y_pred.shape: ',y_pred.shape)
    #     y_margin_pred = y_pred

    #     #Compute the loss 
    #     # print('y_margin_batch.shape: ',y_margin_batch.shape)
    #     # raise NotImplementedError
    #     loss = torch.stack([criterion(y_margin_pred[:,i], y_margin_batch[:,i]) for i in range(args.steps_ahead)]).mean()
    #     #loss = criterion(y_margin_pred.squeeze(), y_margin_batch[:,-1].squeeze())

    #     #If we are actually training the model instead of performing validation
    #     if training:

    #         #Backpropagation
    #         loss.backward()

    #         #Optimizer step
    #         optimizer.step() 
    #         optimizer.zero_grad()

    #     #Track the batch loss
    #     temp_loss += loss.item()

    # #Take the average loss over the batches
    # return temp_loss / len(loader)

#Makes the predictions
def predictions(args, dataset, model):
    """
    Performs the model evalutation over the pytorch dataset

    Parameters
    ----------
    - dataset : `pytorch.Dataset`
        Dataset over which we want to base our prediction
    - model : `pytorch.nn.Module`
        model we want to use for the prediction
    """

    #Define a dataloader from the dataset without 
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    #Define two tensors
    y_margin_preds = torch.zeros(len(dataloader), args.steps_ahead)
    y_margin_trues = torch.zeros(len(dataloader), args.steps_ahead)
    y_benchmark_trues = torch.zeros(len(dataloader), args.steps_filename)

    loop = tqdm(dataloader, desc='Prediction')
    
    for i_sample, (X, y_margin, r, y_benchmark) in enumerate(loop):
        with torch.no_grad():
            
            y_preds = model.forward_pred(X,r).cpu()
            # FOCUS ON FIXED LEG
            y_margin_preds[i_sample] = y_preds
            y_margin_trues[i_sample]  = y_margin.cpu()
            y_benchmark_trues[i_sample]  = y_benchmark.cpu()
            
    return y_margin_trues.numpy(), y_benchmark_trues.numpy(), y_margin_preds.numpy()

class LSTMHiddenExtractor(nn.Module):

           def __init__(self):
               super(LSTMHiddenExtractor, self).__init__()

           def forward(self, x):
               tensor, _ = x
               return tensor

class LSTMPermuter(nn.Module):

           def __init__(self):
               super(LSTMPermuter, self).__init__()

           def forward(self, x):
               tensor = x.permute(0, 2, 1)
               return tensor
           
class Regressor(nn.Module):
     
    def __init__(self, args):
         super(Regressor, self).__init__()

         self.regressor = nn.Sequential(nn.ReLU(),
                                        nn.Linear(args.lstm_hidden_size + 1, args.regressor_hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(args.regressor_hidden_size_2 if args.number_regressor_layers > 1 else args.regressor_hidden_size, 1),
         )

    def forward(self,lstm_hidden,r):

        return self.regressor(torch.squeeze(torch.cat([lstm_hidden[:,-1,:], r],dim=1)))