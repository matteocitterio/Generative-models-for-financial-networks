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

def GetSimAndCIR(args):
    """
    Produces a sim and CIRProcess object
    """

    # Generate CIR process & simulation
    sim = Simulation(args.alpha, args.b, args.sigma, args.v_0, args.years, gamma = args.gamma, seed=True)
    CIRProcess = torch.tensor(sim.CIRProcess.reshape(-1,1)).to(torch.float32).to(device)
    
    return sim, CIRProcess

def GetContracts(args, sim):
    """
    Create contracts, returns contracts and maximum number of active contracts
    """
    #Generate arrival times for the contracts
    arrival_times = np.array([i * args.contract_frequency for i in range(args.years) if i * args.contract_frequency < 365* args.years])

    #Build the contracts using Simulation and arrival times
    contracts = utils.CreateContracts(arrival_times, sim)

    # Compute maximum number of simultaneously active contracts
    max_n_active_contracts = utils.GetMaximumNActiveContracts(contracts, sim)

    return contracts, max_n_active_contracts

def CreateDataset(contracts, sim, CIRProcess, max_n_active_contracts):

    """
    Builds and returns X, y, r as tensors
    """

    #Retrieve the number of feautures per contract
    n_contract_features = len(contracts[0].get_contract_features(0))

    #I will fill this one with the indexis of the active contracts
    X = []
    y_margin = []
    conditioning = []

    for i_time, t in enumerate(range(sim.TotPoints-1)):
        active_contracts = [contract for contract in contracts if contract.is_active(t)]
        
        #This will contain the contracts at a certain time
        temp_contract_row = torch.zeros((max_n_active_contracts*n_contract_features))
        temp_y_margin = 0

        for i_contract, contract in enumerate(active_contracts):
            
            #Retrieve contract features
            contract_features = contract.get_contract_features(t)    
            #Fill the row    
            temp_contract_row[i_contract*len(contract_features) : (i_contract+1)*len(contract_features)] = contract_features
            #Fill the target accordingly
            temp_y_margin += contract.GetVariationMargin(t)
            
        #Append the computed quantities if contract array is different from just zeros
        if not torch.all(torch.eq(temp_contract_row, torch.zeros_like(temp_contract_row))):
            
            X.append(temp_contract_row)
            y_margin.append(temp_y_margin)
            conditioning.append(CIRProcess[i_time])

    #Transform quantities in tensors
    X = torch.vstack(X)
    y_margin= torch.tensor(y_margin)
    conditioning = torch.tensor(conditioning)

    return X, y_margin, conditioning

def GetDataloader(X, y_margin, conditioning):
    # Train-test split
    training_index = int(0.8 * X.shape[0])

    #TRAIN
    y_margin_train = y_margin[:training_index]
    X_train = X[:training_index, :]
    CIR_train = conditioning[:training_index]
    #TEST
    y_margin_test = y_margin[training_index:]
    X_test = X[training_index:, :]
    CIR_test = conditioning[training_index:]
        
    # Slice dataset into windows
    lookback = 20
    X_train, y_margin_train, r_train = utils.create_windows(device, X_train, y_margin_train, CIR_train, lookback=lookback)
    X_test,y_margin_test, r_test = utils.create_windows(device, X_test, y_margin_test, CIR_test, lookback=lookback)

    #Build the pytorch datasets and dataloaders
    train_dataset = TensorDataset(X_train.to(device), y_margin_train.to(device), r_train.to(device))
    test_dataset = TensorDataset(X_test.to(device), y_margin_test.to(device), r_test.to(device))

    torch.manual_seed(0)
    return DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size), DataLoader(test_dataset, shuffle=True, batch_size = args.batch_size)

def DatasetBuilding(args):

    sim, CIRProcess = GetSimAndCIR(args)
    contracts, max_n_active_contracts = GetContracts(args, sim)
    X, y_margin, conditioning = CreateDataset(contracts, sim, CIRProcess, max_n_active_contracts)
    return GetDataloader(X, y_margin, conditioning)



