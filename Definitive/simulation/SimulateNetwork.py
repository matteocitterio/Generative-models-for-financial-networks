from simulation import Simulation
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.data import Data

#Get command line input as described in the Readme.md
parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--do_simulation', action='store_true', help='Whether to do the simulation or to load a pre-existing sim.E matrix')
parser.add_argument('--device', type=str, help='Device')
parser.add_argument('--nodes', type=int, help='Number of nodes in the network')
parser.add_argument('--steps', type=int, help='Steps ahead')

"""
Example usage:

    `python SimulateNetwork.py --do_simulation --device cuda --nodes 10 --steps 1`

"""

args = parser.parse_args()
device = args.device

scaler = MinMaxScaler()

# Define some default parameters for the Simulation
alpha = 0.6
b = 0.04
sigma = 0.14
v_0 = 0.04
gamma = 3
years = 60
num_nodes = args.nodes

print(f'Doing simulation of {args.nodes} nodes, {years} years, {args.steps} steps ahead')

#Instantiate simulation (using seed = True)
sim = Simulation(alpha, b, sigma, v_0, years, seed = True, num_nodes = num_nodes, gamma = gamma)
CIRProcess = torch.tensor(sim.CIRProcess.reshape(-1,1)).to(torch.float32).to(device)

if args.do_simulation:

    #Run the actual simulation
    sim.SimulateAllEdges(steps_ahead = args.steps)
    np.save('../data/edge_features.npy', sim.E)

else:
    
    #Load the matrix
    sim.E = np.load('../data/edge_features.npy', allow_pickle=True)


# Get the simulation active contracts
active_contracts, active_days = sim.GetActiveContractList()
print('len of active: ', len(active_contracts), len(active_days))

#Get the maximum number of simultaneously active contracts (it gives the shape of the tensor)
max_n_active_contracts = sim.GetMaximumNActiveContracts(active_contracts)
print(f"Max number of simultaneously active contracts: {max_n_active_contracts}")

#Get contract size
contract_size = len(active_contracts[0][0].get_contract_features(active_days[0]) )   

#Define a tqdm loop for UI friendliness
loop = tqdm(range(len(active_contracts)), desc='Time')
dataset = []

h=0

y_benchmark = torch.zeros(((len(active_contracts), args.nodes, args.steps)))

for t in loop:

    #Node feature matrix
    X = torch.zeros((num_nodes, contract_size * max_n_active_contracts)).to(device)
    #Node target array
    y = torch.zeros((num_nodes)).to(device)

    #Set the current advancement
    loop.set_postfix(t=t)


    #Retrieve (src,dst) for active contracts at time t and build a tensor of shape (2, |\mathcal(E)|)
    edges = torch.stack([torch.tensor([contract.src, contract.dst]) for contract in active_contracts[t]], dim=1)

    #This selects only the edges that actually have a contract, unregarding of the number of contracts they have
    edge_index, unique_indices, counts = torch.unique(edges.cpu(), dim=1, return_inverse=True, return_counts=True)
    
    #Convert edge_index to int type
    edge_index = edge_index.to(torch.int64)

    #This selects the source nodes for active contracts at time time
    source_nodes = torch.unique(edge_index[0,:])

    #Cycle over X rows
    #Here we consider node i, X_i
    for i in source_nodes:
        
        #take active contracts just for node i
        contracts_for_node_i = [contract for contract in active_contracts[t] if contract.src==i]

        for eta in range(args.steps):
            
            #Compute the benchmark to compare with the model's results
            y_benchmark[h,i, eta] = sim.ProvideBenchmark(t_l=t, steps_ahead=eta+1, contracts=contracts_for_node_i, n_simulations=100)

        for i_contract, contract in enumerate(contracts_for_node_i):

            #Compute M_ij
            try:
                y[i] += contract.GetVariationMargin(active_days[t])
                contract_features = contract.get_contract_features(active_days[t])                  

            except:
                #Error handling
                print('EXCEPTION OCCURED')
                print('t:',t)
                print('contract: ',contract)
                print('contract.is_active(t): ',contract.is_active(t))
                print('active_days[t]: ', active_days[t])
                print('contract.is_active(active_days[t]): ',contract.is_active(active_days[t]))
                raise NotImplementedError   
            
            try:
                X[i,i_contract*len(contract_features) : (i_contract+1)*len(contract_features)] = contract_features
            except:
                #Error handling
                print('EXCEPTION OCCURED')
                print('t:',t)
                print('Contracts for node i: ', contracts_for_node_i)
                print('contract: ',contract)
                print('active_days[t]: ', active_days[t])
                print('i: ', i)
                print('X[i]: ', i)
                print('i_contract: ', i_contract )
                print('i_contract*len(contract_features):', i_contract*len(contract_features))
                print('(i_contract+1)*len(contract_features): ', (i_contract+1)*len(contract_features))
                print('contract_features: ', contract_features)
                raise NotImplementedError   

    h+=1

    subgraph = Data(edge_index=edge_index, x = X, y = y, r=CIRProcess[active_days[t]],node_feat=sim.node_features, num_nodes = num_nodes)
    dataset.append(subgraph)

#Save the dataset
torch.save(y_benchmark,f'../data/y_benchmark_{num_nodes}nodes_beta.pt')
torch.save(dataset, f'../data/subgraphs_Duffie_{num_nodes}nodes_{sim.gamma}gamma_TEST.pt')