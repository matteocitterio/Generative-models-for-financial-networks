from simulation import Simulation, GetActiveContractsIndices, GetSimulationQuantitiesTensors, return_max_num_contracts, scale_feature_matrix, scale_targets
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.transforms import one_hot_degree, Compose, ToDevice
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data
from torch_geometric.data.temporal import TemporalData
from torch.utils.data import Dataset
import torch.nn.functional as F

#Get command line input
parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--do_simulation', action='store_true', help='Whether to do the simulation or to load a pre-existing sim.E matrix')
parser.add_argument('--device', type=str, help='Device')
parser.add_argument('--nodes', type=int, help='Number of nodes in the network')
parser.add_argument('--gamma', type=float, help='Gamma for the arrival times process')

args = parser.parse_args()
device = args.device

scaler = MinMaxScaler()

# Define some default parameters for the Simulation
alpha = 0.6
b = 0.02
sigma = 0.14
v_0 = 0.04
years = 20
num_nodes = args.nodes

print(f'Doing simulation of {args.nodes} nodes and \gamma = {args.gamma}')

#Instantiate simulation (using seed = True)
sim = Simulation(alpha, b, sigma, v_0, years, seed = True, num_nodes = num_nodes, gamma = args.gamma)

if args.do_simulation:

    #Run the actual simulation
    sim.SimulateAllEdges()
    np.save('/u/mcitterio/data/edge_features.npy', sim.E)

else:
    sim.E = np.load('/u/mcitterio/data/edge_features.npy', allow_pickle=True)

# Get the simulation contracts as tensors
contracts = GetSimulationQuantitiesTensors(sim, device)

# Now we need to get the list of indexis of active contracts at time t:
active_contracts_indices = GetActiveContractsIndices(sim, contracts)

# Get the maximum number of simultaneously active contracts observed for an edge throughout the entire horizon [0,T]
max_num_contracts = return_max_num_contracts(contracts, active_contracts_indices)

#Define a tqdm loop for UI friendlyness
loop = tqdm(range(len(active_contracts_indices)-1), desc='Time')
dataset = []

for t in loop:

    #Node feature matrix
    X = torch.zeros((num_nodes, 5 * max_num_contracts)).to(device)
    #Node target array
    y = torch.zeros((num_nodes)).to(device)

    #Set the current advancement
    loop.set_postfix(t=t)

    #Index are the indices for active contracts at time t
    index = active_contracts_indices[t]
    #Actually active contracts at this time
    active_contract_at_time_t = contracts[index]
    
    #This are the edge indexis for all the contracts
    edge_index = torch.stack([active_contract_at_time_t[:,0], active_contract_at_time_t[:,1]], dim=0).to(device)
    #This selects only the edges that actually have a contract, unregarding of the number of contracts they have
    edge_index, unique_indices, counts = torch.unique(edge_index.cpu(), dim=1, return_inverse=True, return_counts=True)
    #Convert edge_index to int type
    edge_index = edge_index.to(torch.int64)
    
    #This selects the source nodes for active contracts at time time
    source_nodes = active_contract_at_time_t[:,0].to(int)
    unique_source_nodes, _ = torch.unique(source_nodes.cpu(), return_inverse=True) 

    time_array = contracts[:,2]

    #Cycle over X rows
    for i in unique_source_nodes:

        #Here we consider node i, X_i
        
        #take active contracts just for node i
        indices_for_node_i = (source_nodes == i).nonzero(as_tuple=True)
        contracts_for_node_i = active_contract_at_time_t[indices_for_node_i]

        #Take as target M^i(t)
        y[i] = sim.GetMtForNode(i, edge_index, t)
    
        #Cycle over active contracts for node i
        for j in range(contracts_for_node_i.shape[0]):

            #Fill feature matrix accordingly
            #Subtract in order to obtain time to maturity
            contracts_for_node_i[j][3] = contracts_for_node_i[j][3] #- t
            X[i][0 + (j*5) : 5 + (j*5)] = contracts_for_node_i[j][2:7]

    subgraph = Data(edge_index=edge_index, x = X, y = y, num_nodes = num_nodes)
    dataset.append(subgraph)

torch.save(dataset, f'/u/mcitterio/data/subgraphs_Duffie_{num_nodes}nodes_{sim.gamma}gamma.pt')

# Load the list of Data objects
#loaded_subgraphs = torch.load('subgraphs.pt')